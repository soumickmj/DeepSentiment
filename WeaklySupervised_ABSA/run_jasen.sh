# The code is completely from the weakly supervised code https://github.com/teapot123/JASen
# dataset directory
dataset=restaurant
#dataset=laptop

# For which convolutional layer to use XSepConv (1 means use)
xsepconv='0 0 1'

# text file name; one document per line
text_file=train.txt

# keyword file
topic_file1=senti_w_kw.txt
topic_file2=aspect_w_kw.txt

topic=mix

# load pretrained word2vec embedding
pretrain_emb=word2vec_100.txt
#pretrain_emb=glove_twitter_27B_100d.txt
#pretrain_emb=my_cc.en.100.vec

if [ ! -f "$pretrain_emb" ] && [ -f "word2vec_100.zip" ]; then
#if [ ! -f "$pretrain_emb" ] && [ -f "glove_twitter_27B_100d.zip" ]; then
#if [ ! -f "$pretrain_emb" ] && [ -f "my_cc.en.100.zip" ]; then
    echo "Unzipping downloaded pretrained embedding"
    unzip word2vec_100.zip && rm word2vec_100.zip
    #unzip my_cc.en.100.zip && rm my_cc.en.100.zip
    #unzip glove_twitter_27B_100d.zip && rm glove_twitter_27B_100d.zip
fi

cd src
make joint
cd ..

./src/joint -train ./datasets/${dataset}/${text_file} \
	-topic1-name ./datasets/${dataset}/${topic_file1} \
	-topic2-name ./datasets/${dataset}/${topic_file2} \
	-load-emb ${pretrain_emb} \
	-spec ./datasets/${dataset}/emb_${topic}_spec.txt \
	-res ./datasets/${dataset}/res_${topic}.txt -k 10 -expand 1 \
	-word-emb ./datasets/${dataset}/emb_${topic}_w.txt \
	-doc ./datasets/${dataset}/emb_${topic}_d.txt \
	-topic-emb ./datasets/${dataset}/emb_${topic}_t.txt \
	-size 100 -window 5 -negative 5 -sample 1e-3 -min-count 2 \
	-threads 10 -binary 0 -iter 5 -pretrain 2 -global_lambda 2.5
		
cd src
make margin
cd ..

topic_file=senti_w_kw.txt

topic=$(echo ${topic_file} | cut -d'.' -f 1)

./src/margin -train ./datasets/${dataset}/${text_file} \
		-topic-name ./datasets/${dataset}/${topic_file} \
		-spec ./datasets/${dataset}/emb_${topic}_spec.txt \
		-context ./datasets/${dataset}/emb_${topic}_v.txt \
		-res ./datasets/${dataset}/res_${topic}.txt -k 10 -expand 1 \
		-word-emb ./datasets/${dataset}/emb_${topic}_w.txt \
		-topic-emb ./datasets/${dataset}/emb_${topic}_t.txt \
		-size 100 -window 5 -negative 5 -sample 1e-3 -min-count 2 \
		-threads 20 -binary 0 -iter 5 -pretrain 2 -global_lambda 2.5

topic_file=aspect_w_kw.txt

topic=$(echo ${topic_file} | cut -d'.' -f 1)

./src/margin -train ./datasets/${dataset}/${text_file} \
		-topic-name ./datasets/${dataset}/${topic_file} \
		-spec ./datasets/${dataset}/emb_${topic}_spec.txt \
		-context ./datasets/${dataset}/emb_${topic}_v.txt \
		-res ./datasets/${dataset}/res_${topic}.txt -k 10 -expand 1 \
		-word-emb ./datasets/${dataset}/emb_${topic}_w.txt \
		-topic-emb ./datasets/${dataset}/emb_${topic}_t.txt \
		-size 100 -window 5 -negative 5 -sample 1e-3 -min-count 2 \
		-threads 20 -binary 0 -iter 5 -pretrain 2 -global_lambda 2.5

python evaluate.py --dataset datasets/${dataset}



