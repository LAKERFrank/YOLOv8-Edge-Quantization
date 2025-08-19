```


docker build --no-cache -t tracknetv4 .
docker build -t tracknetv4 .

docker run --gpus all -it tracknetv4
docker run --gpus all --ipc=host -v /hdd/dataset/tracknetv4:/usr/src/datasets/tracknet -it tracknetv4
docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_4 \
-v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_5 \
-v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_6 \
-v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_7 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_8 \
-v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_9 \
-v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_10 \
-v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_11 \
-v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_12 \
-v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/train_data/profession_match_13 \
-v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/train_data/profession_match_14 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_15 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/val_data/profession_match_21 \
-v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/val_data/profession_match_22 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4

docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/blion_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/lamsopoor_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/blion_tracknet:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4

docker exec -it 2571d5854ec3bdaae47f9c43a6f8e80e362196bf7a52acaa56b15143715f290f /bin/bash

docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/val_data/profession_match_1 \
-it tracknetv4



docker run -v /hdd/datasets:/usr/src/datasets --gpus all -it tracknetv4

docker run -v C:\Users\user1\bartek\github\BartekTao\datasets:/usr/src/datasets -it tracknetv4

git fetch
git checkout feat/non_sigmoid_predict

git fetch
git checkout feat/sigmoid_pos

python tracknet.py --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 3

python tracknet.py  --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 70

python tracknet.py --batch 1  --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 50

python tracknet.py --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 100

python tracknet.py --model_path /usr/src/ultralytics/runs/detect/train226/weights/best.pt --epoch 1


python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train196/weights/best.pt --source /usr/src/datasets/tracknet/val_data

```

```
docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_4 \
-v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_5 \
-v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_6 \
-v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_7 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_8 \
-v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_9 \
-v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_10 \
-v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_11 \
-v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_12 \
-v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/train_data/profession_match_13 \
-v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/train_data/profession_match_14 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_15 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/val_data/profession_match_3 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/val_data/profession_match_8 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4

docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/val_data/profession_match_3 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4



docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_4 \
-v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_5 \
-v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_6 \
-v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_7 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_8 \
-v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_9 \
-v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_10 \
-v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_11 \
-v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_12 \
-v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/train_data/profession_match_13 \
-v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/train_data/profession_match_14 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_16 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_17 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_18 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_19 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_20 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_21 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_22 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_23 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_24 \
-v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/train_data/profession_match_25 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/val_data/profession_match_26 \
-v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/val_data/profession_match_27 \
-v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/val_data/profession_match_28 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4


docker run --gpus all --ipc=host \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_4 \
-v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_5 \
-v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_6 \
-v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_7 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_8 \
-v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_9 \
-v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_10 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/val_data/profession_match_21 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4

docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/alex_tracknet/.cache:/usr/src/datasets/tracknet/train_data/.cache \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/test_data/match_1:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it tracknetv4

docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/test_data/match_1:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-it pony830906/tracknet-v4:v1.0.0-beta

## 測試 pt 是否有正確儲存
docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/test_data/match_1:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-it tracknetv4

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train196/weights/best.pt --source /usr/src/datasets/tracknet/val_data/profession_match_20

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train226/weights/best.pt --source /usr/src/datasets/tracknet/test_data/tracknetv4_test_data

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train226/weights/best.pt --source /usr/src/datasets/tracknet/train_data/profession_match_1

ksrg owxb gtoq aokx

# 20240929
docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
-it tracknetv4

# 20241229
docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
-it tracknetv4

# 20250113
docker run --gpus all --ipc=host \
-v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
-v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/profession_match_2 \
-v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/profession_match_3 \
-v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/profession_match_4 \
-v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_5 \
-v /hdd/dataset/profession_match_2:/usr/src/datasets/tracknet/train_data/profession_match_6 \
-v /hdd/dataset/profession_match_3:/usr/src/datasets/tracknet/train_data/profession_match_7 \
-v /hdd/dataset/profession_match_4:/usr/src/datasets/tracknet/train_data/profession_match_8 \
-v /hdd/dataset/profession_match_5:/usr/src/datasets/tracknet/train_data/profession_match_9 \
-v /hdd/dataset/profession_match_6:/usr/src/datasets/tracknet/train_data/profession_match_10 \
-v /hdd/dataset/profession_match_7:/usr/src/datasets/tracknet/train_data/profession_match_11 \
-v /hdd/dataset/profession_match_8:/usr/src/datasets/tracknet/train_data/profession_match_12 \
-v /hdd/dataset/ces2025_all_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
-v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
-v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
-v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
-v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
-v /hdd/dataset/tracknetv4/.cache:/usr/src/datasets/tracknet/train_data/.cache \
-it tracknetv4

python tracknet.py --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 200

python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 20 &

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train238/weights/best.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train253/weights/best.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode predict --batch 1 --model_path /usr/src/ultralytics/runs/detect/train264/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train264/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train266/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train267/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train288/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train302/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val --batch 1 --model_path /usr/src/ultralytics/runs/detect/train322/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val_v1 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train328/weights/last.pt --source /usr/src/datasets/tracknet/train_data

python tracknet.py --mode val_v1 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train341/weights/last.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train345/weights/last.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train431/weights/epoch120.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train437/weights/epoch50.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train442/weights/epoch20.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train443/weights/epoch30.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train475/weights/best.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train618/weights/last.pt --source /usr/src/datasets/tracknet/val_data

python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train618/weights/best.pt --source /usr/src/datasets/tracknet/val_data/profession_match_1_test/frame/1_06_09/

python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train642/weights/best.pt --epoch 20 &
python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train540/weights/last.pt --epoch 200 &
python tracknet.py --mode train_v3 --model_path /usr/src/ultralytics/runs/detect/train618/weights/best.pt --epoch 50 &
python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train630/weights/last.pt --epoch 20 &

/hdd/dataset/alex_tracknet/frame/1_05_07/
192
192/20 = 9...12
feat/add_hit_label_without_dxdxloss

C:\Users\user1\bartek\github\BartekTao\ultralytics\runs\detect\train345\weights\last.pt

{'pos_FN': 13368, 'pos_FP': 0, 'pos_TN': 0, 'pos_TP': 901, 'pos_acc': 0.06314387833765506, 'pos_precision': 1.0, 'conf_FN': tensor(14180, device='cuda:0'), 'conf_FP': tensor(17994, device='cuda:0'), 'conf_TN': tensor(7639737, device='cuda:0'), 'conf_TP': tensor(89, device='cuda:0'), 'conf_acc': tensor(0.9958, device='cuda:0'), 'conf_precision': tensor(0.0049, device='cuda:0'), 'threshold>0.8 rate': tensor(904.1500, device='cuda:0')}


4000 個點
10 顆球

```