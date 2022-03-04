wget "https://cmsc848q-hw2-group11.s3.amazonaws.com/tfidf.pickle"
mkdir models
mv tfidf.pickle models/

wget "https://cmsc848q-hw2-group11.s3.amazonaws.com/answer_extractor.zip"
unzip answer_extractor.zip
mv answer_extractor/ models/

wget "https://cmsc848q-hw2-group11.s3.amazonaws.com/custom_data.zip"
unzip custom_data.zip