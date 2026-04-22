Bug 分析
Bug 1：__getitem__ 原地修改了 self.train_data（严重）
gembench_dataset.py:89-92
def __getitem__(self, idx):
    sample=self.train_data[idx]
    sample["lang_goal"] = random.choice(sample["lang_goal"])   # randomly choose one instruction for every fetching. This is important for generalization.
    return sample
根因：sample = self.train_data[idx] 是引用，不是拷贝。sample["lang_goal"] = random.choice(...) 直接改写了 self.train_data[idx]["lang_goal"]：

第 1 个 epoch：lang_goal 是 list，random.choice(list) → 返回一条指令字符串，然后把字符串写回了 self.train_data[idx]
第 2 个 epoch 起：lang_goal 已经变成了一个字符串，random.choice(string) → 从字符串里随机抽一个字符（如 "p"、"u"……），lang_goal 彻底损坏
Bug 2：DistributedSampler 未调用 set_epoch()（每个 epoch shuffle 顺序完全相同）
train.py:370-376
while True:
    if i == end_epoch:
        break
 
    print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")
 
    out = train(agent, train_dataloader, rank=dist.get_rank(),cameras=cmd_args.cameras)
DistributedSampler 内部用 self.epoch（默认 0）作为随机种子一部分。若不调用 sampler.set_epoch(i)，self.epoch 永远是 0，每个 epoch 的 shuffle 顺序完全一样。PyTorch 官方文档明确要求在每个 epoch 开始前调用 set_epoch()。

