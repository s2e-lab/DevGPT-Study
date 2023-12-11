class CustomSFTTrainer(SFTTrainer):
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = self._get_train_sampler()

        def dynamic_padding_collate_fn(examples):
            # Compute max length in this batch and truncate/pad all examples to this length.
            max_length_in_batch = min(max(len(example) for example in examples), self.args.max_seq_length)
            return [example[:max_length_in_batch] for example in examples]

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=dynamic_padding_collate_fn,
            drop_last=self.args.dataloader_drop_last,
        )
