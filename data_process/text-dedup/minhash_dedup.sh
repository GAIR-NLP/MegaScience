python minhash_dedup.py \
  --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/refine_qa_with_document/original_data_96w.jsonl" \
  --local \
  --split "train" \
  --cache_dir "./cache" \
  --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/refine_qa_with_document/original_data_dedup.jsonl" \
  --column "question" \
  --batch_size 10000 \
  --use_auth_token true \
  --b 1 \
  --r 10 \
  --ngram 3 \
  --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/economics.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/economics" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/mathpile.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/mathpile" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/medicine.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/medicine" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/physics.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/physics" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/math.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/math" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6


# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/cs.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/cs" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/chemistry.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/chemistry" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6

# python minhash_dedup.py \
#   --path "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/original_qa_with_document/biology.jsonl" \
#   --local \
#   --split "train" \
#   --cache_dir "./cache" \
#   --output "/inspire/hdd/global_user/liupengfei-24025/rzfan/scitextbooks_extracted_qa/dedup_qa_with_document/biology" \
#   --column "question" \
#   --batch_size 10000 \
#   --use_auth_token true \
#   --b 1 \
#   --r 10 \
#   --ngram 3 \
#   --threshold 0.6