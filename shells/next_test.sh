GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=nextqa \
	--dataset=nextqa \
	--mc=5 \
	--bnum=10 \
	--test=1 \
	--qmax_words=0 \
	--amax_words=38 \
	--max_feats=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=4 \
	--mlm_prob=0 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--lan="RoBERTa" \
	--save_dir='../data/save_models/nextqa/CoVGT_FTCoWV/' \
	--pretrain_path='../data/save_models/nextqa/CoVGT_FTCoWV/best_model.pth' \
	#--CM_PT=1
