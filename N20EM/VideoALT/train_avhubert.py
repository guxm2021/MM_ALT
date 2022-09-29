#!/usr/bin/env/python3
import os
import sys
import torch
import logging
import numpy as np
import speechbrain as sb
from speechbrain.utils.distributed import run_on_main
import utils as custom_utils
from hyperpyyaml import load_hyperpyyaml
from pathlib import Path
logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        # load video
        video, vid_lens = batch.video
        # load text
        tokens_bos, _ = batch.tokens_bos
        video, vid_lens = video.to(self.device), vid_lens.to(self.device)
        video = video.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T, H, W]

        # Forward pass
        video_feats = {"video": video, "audio": None}
        video_feats = self.modules.wav2vec2(video_feats)   # [B, T, C]

        # encoder-decoder ASR architecture
        x = self.modules.enc(video_feats)

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(x)
        p_ctc = self.hparams.log_softmax(logits)

        e_in = self.modules.emb(tokens_bos)
        h, _ = self.modules.dec(e_in, x, vid_lens)

        # output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(h)
        p_seq = self.hparams.log_softmax(logits)

        # Compute outputs
        if stage == sb.Stage.VALID:
            hyps = sb.decoders.ctc_greedy_decode(
                p_ctc, vid_lens, blank_id=self.hparams.blank_index
            )
            # self.modules.lm_model.to(self.device)
            # hyps, scores = self.hparams.greedy_searcher(x, vid_lens)
            return p_ctc, p_seq, vid_lens, hyps

        elif stage == sb.Stage.TEST:
            if self.hparams.save_feat:
                hyps = sb.decoders.ctc_greedy_decode(
                       p_ctc, vid_lens, blank_id=self.hparams.blank_index
                )
                # save features
                torch.save(video_feats[0].detach().cpu(), os.path.join(self.hparams.save_feat_folder, batch.id[0] + ".pt"))
            else:
                self.modules.lm_model.to(self.device)
                hyps, scores = self.hparams.beam_searcher(x, vid_lens)
            return p_ctc, p_seq, vid_lens, hyps

        return p_ctc, p_seq, vid_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        if stage == sb.Stage.TRAIN:
            p_ctc, p_seq, vid_len = predictions
        else:
            p_ctc, p_seq, vid_len, hyps = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, vid_len, tokens_lens)
        loss_seq = self.hparams.seq_cost(p_seq, tokens_eos, tokens_eos_lens)
        loss = self.hparams.ctc_weight * loss_ctc
        loss += (1 - self.hparams.ctc_weight) * loss_seq

        if stage != sb.Stage.TRAIN:
            # Decode token terms to words
            predicted_words = [
                "".join(self.tokenizer.decode_ndim(utt_seq)).split(" ")
                for utt_seq in hyps
            ]
            # predicted_words = [
            #     "".join(self.tokenizer.decode_ids(utt_seq)).split(" ")
            #     for utt_seq in hyps
            # ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]
            self.ctc_metrics.append(ids, p_ctc, tokens, vid_len, tokens_lens)
            self.seq_metrics.append(ids, p_seq, tokens_eos, tokens_eos_lens)
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # Managing automatic mixed precision
        if self.auto_mix_prec:
            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.wav2vec_optimizer)
            self.scaler.unscale_(self.adam_optimizer)

            if self.check_gradients(loss):
                self.scaler.step(self.wav2vec_optimizer)
                self.scaler.step(self.adam_optimizer)

            self.scaler.update()
        else:
            predictions = self.compute_forward(batch, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

            loss.backward()
            if self.check_gradients(loss):
                self.wav2vec_optimizer.step()
                self.adam_optimizer.step()

            self.wav2vec_optimizer.zero_grad()
            self.adam_optimizer.zero_grad()

        return loss.detach().cpu()
    
    def on_evaluate_start(self, max_key=None, min_key=None):
        """Gets called at the beginning of ``evaluate()``

        Default implementation loads the best-performing checkpoint for
        evaluation, based on stored metrics.

        Arguments
        ---------
        max_key : str
            Key to use for finding best checkpoint (higher is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        min_key : str
            Key to use for finding best checkpoint (lower is better).
            By default, passed to ``self.checkpointer.recover_if_possible()``.
        """
        # Recover best checkpoint for evaluation
        if self.checkpointer is not None:
            self.checkpointer.recover_if_possible(
                max_key=max_key,
                min_key=min_key,
                device=torch.device(self.device),
            )
        # save wav2vec 2.0 and encoder-decoder model
        if self.hparams.save_model:
            os.makedirs(self.hparams.save_model_folder, exist_ok=True)
            torch.save(self.hparams.model.state_dict(), os.path.join(self.hparams.save_model_folder, 'model.pt'))
            logger.info(f"Save wav2vec 2.0 and encoder-decoder model to the folder: {self.hparams.save_model_folder}")
        else:
            logger.info("No wav2vec 2.0 and encoder-decoder model to save")

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        predictions = self.compute_forward(batch, stage=stage)
        loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        self.ctc_metrics = self.hparams.ctc_stats()
        self.seq_metrics = self.hparams.seq_stats()
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr_model, new_lr_model = self.hparams.lr_annealing_adam(
                stage_stats["WER"]
            )
            old_lr_wav2vec, new_lr_wav2vec = self.hparams.lr_annealing_wav2vec(
                stage_stats["WER"]
            )
            sb.nnet.schedulers.update_learning_rate(
                self.adam_optimizer, new_lr_model
            )
            sb.nnet.schedulers.update_learning_rate(
                self.wav2vec_optimizer, new_lr_wav2vec
            )
            self.hparams.train_logger.log_stats(
                stats_meta={
                    "epoch": epoch,
                    "lr_model": old_lr_model,
                    "lr_wav2vec": old_lr_wav2vec,
                },
                train_stats=self.train_stats,
                valid_stats={
                    "loss": stage_loss,
                    "ctc_loss": self.ctc_metrics.summarize("average"),
                    "seq_loss": self.seq_metrics.summarize("average"),
                    "CER": stage_stats["CER"],
                    "WER": stage_stats["WER"],
                },
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

    def init_optimizers(self):
        "Initializes the wav2vec2 optimizer and model optimizer"
        self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
            self.modules.wav2vec2.parameters()
        )
        self.adam_optimizer = self.hparams.adam_opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable(
                "wav2vec_opt", self.wav2vec_optimizer
            )
            self.checkpointer.add_recoverable("adam_opt", self.adam_optimizer)


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="frame")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="frame", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="frame")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="frame"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # 3. Define video pipeline:
    image_crop_size = 88
    image_mean = 0.421
    image_std = 0.165
    transform_train = custom_utils.Compose([
                      custom_utils.Normalize( 0.0,255.0 ),
                      custom_utils.RandomCrop((image_crop_size, image_crop_size)),
                      custom_utils.HorizontalFlip(0.5),
                      custom_utils.Normalize(image_mean, image_std) ])

    transform_eval  = custom_utils.Compose([
                      custom_utils.Normalize( 0.0,255.0 ),
                      custom_utils.CenterCrop((image_crop_size, image_crop_size)),
                      custom_utils.Normalize(image_mean, image_std) ])
    # custom_utils.Compose([
    #                   custom_utils.Normalize( 0.0,255.0 ),
    #                   custom_utils.RandomCrop((image_crop_size, image_crop_size)),
    #                   custom_utils.Normalize(image_mean, image_std) ])

    @sb.utils.data_pipeline.takes("mp4")
    @sb.utils.data_pipeline.provides("video") 
    def video_pipeline_train(mp4):
        video = custom_utils.load_video(mp4)
        video = transform_train(video)
        video = np.expand_dims(video, axis=-1)
        video = torch.from_numpy(video.astype(np.float32)) # (T, H, W, C)
        return video
    
    @sb.utils.data_pipeline.takes("mp4")
    @sb.utils.data_pipeline.provides("video") 
    def video_pipeline_eval(mp4):
        video = custom_utils.load_video(mp4)
        video = transform_eval(video)
        video = np.expand_dims(video, axis=-1)
        video = torch.from_numpy(video.astype(np.float32)) # (T, H, W, C)
        return video
    
    train_datasets = [train_data]
    eval_datasets = [valid_data] + [i for k, i in test_datasets.items()]
    sb.dataio.dataset.add_dynamic_item(train_datasets, video_pipeline_train)
    sb.dataio.dataset.add_dynamic_item(eval_datasets, video_pipeline_eval)

    # 3. Define text pipeline:
    label_encoder = sb.dataio.encoder.CTCTextEncoder()

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "char_list", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        char_list = list(wrd)
        yield char_list
        tokens_list = label_encoder.encode_sequence(char_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    lab_enc_file = hparams["label_encoder"]
    if os.path.exists(lab_enc_file):
        print("label encoder has already created")
    special_labels = {
        "bos_label": hparams["bos_index"],
        "eos_label": hparams["eos_index"],
        "blank_label": hparams["blank_index"],
    }
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[train_data],
        output_key="char_list",
        special_labels=special_labels,
        sequence_input=True,
    )

    if hparams["save_feat"]:
        os.makedirs(hparams["save_feat_folder"], exist_ok=True)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "video", "wrd", "char_list", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, label_encoder


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_datasets, label_encoder = dataio_prepare(
        hparams
    )

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = label_encoder

    # Training
    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing
    for k in test_datasets.keys():  # keys are test_clean, test_other etc
        asr_brain.hparams.wer_file = os.path.join(
            hparams["output_folder"], "wer_{}.txt".format(k)
        )
        asr_brain.evaluate(
            test_datasets[k], 
            min_key="WER",        # changed by anonymous, use min_key "WER" to load checkpointer
            test_loader_kwargs=hparams["test_dataloader_opts"]
        )