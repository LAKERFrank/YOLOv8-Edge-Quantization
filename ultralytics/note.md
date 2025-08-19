yolov8
    ultralytics\nn\modules\head.py
        調整 detect 輸出，減少長寬以及增加x,y變量

    loss function 修改

    ultralytics\yolo\cfg 修改

    training 的 output weight 要注意 要加重

可以從 dataloader 知道 10 張圖片怎麼丟進去
train_data
    match1
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y
    match2
        video
            xxx.mp4
        frame
            frame<ID>.png
        ball_trajectory
            csv file: frame,x,y


程式碼解析
    nn.module parse_model
        解析 models.yaml 轉換成一層一層的網路，輸入的 channel 可以用參數 ch 帶入
    yaml 的最後一層，會接到 Detect 模型，目前需要修改這邊，讓他可以輸出我要的 10 組(x,y,dx,dy,conf)
    data 的部分怎麼塞進來，也是一大問題

    loss
        FL(pi​,ti​)=−αti​​(1−pi​)γlog(pi​) 球存在
        FL(pi​,ti​)=−αti​​piγ​log(1−pi​)   球不存在

## idea
- 每個網格的x,y 大小會有限制 (需要Sigmoid)
- from yoloX: 可以把 cell 中心四周的 cell (3*3)也當作 positive 去算 loss
- 是否可以使用 upsampling 做一些事情
- Kalman Filtering


## train record
- train4: 10/9 使用 non sigmoid predict run 50 epoch
- train1: 嘗試使用 sigmoid(x, y, dx, dy) 只跑了個數的 epoch
- train6: 使用 weight=100
- train49: weight=100, 16 batch, fix focal loss, epoch 50,commit ba2d8ff
- train71: 11/16 tanh, bce loss, epoch 50 commit bd47c9b
- train79: 圖像化，50 epoch，可用來測試 conf loss
- train81: 圖像化，100 epoch，可用來測試 conf loss
- train83: 圖像化，100 epoch，可用來測試 conf loss，包含其他數字
- C:\Users\user1\bartek\github\BartekTao\datasets\tracknet\check_training_img5: conf 修正完成
- train148 加入所有 loss 100 epoch
- train181 pos+conf loss 50 epoch (head不拆分) check_training_img9
- train183 pos+conf+mov loss 47 epoch (head不拆分) check_training_img10
- train196 pos+conf+mov loss 100 epoch (head不拆分) check_training_img11, commit: f1aecf0 (lr=0.01, momentum=0.9)
- train226 整理後架構 100 epoch

feats = 三個 tensor[16*144*80*80],[16*144*40*40],[16*144*20*20] 
no=144
reg_max=16
nc=80

## 20240518
- 在 predict 的時候，dataset 不知道為什麼會變成 len(0)
- 確認 predict 有問題，有可能是模型沒有儲存到，因為我拿 training data 去 predict，但是結果與 training 的時候不同，目前想法有以下:
    - 拿 best.pt 繼續 train，看看結果長什麼樣子，如果異常，代表 model 沒有正常儲存
    - 檢查 predict 程式碼哪裡異常

## 20240519
- 拿模型 pt 進行訓練，看看訓練時的數據，是否符合預期，以確認 pt 是否有正確儲存

## 20240830
- dxdy 有加與沒加入，對於模型學習的能力是否有正向影響

## 20240830
- train238 測試 epoch 200 with dxdy
- coachbox 密碼: nol56680

## docker debug
### 空間不足
```
docker container prune

//查看 /var/lib/docker/ 目錄及其子目錄的大小
sudo du -sh /var/lib/docker/

//執行以下命令來查看包含 /var/lib/docker/ 的分區的總空間及可用空間
df -h /var/lib/docker/
docker exec -it 0c3e1511776e /bin/bash

```

## 20240907
- train253 測試 epoch 200 without dxdy
- Huber Loss 或 Smooth L1 Loss

## 20240917
- train263 epoch 200 使用 DFL without dxdy and hit loss

## 20240919
待辦
- 確認 263 有多少個 > 0.5 的 conf
- 加入 IOU loss (Gaussian IOU)

## 20240920
- train264 epoch 200 iou sigma=0.7

## 20240923
- train266 epoch 200, 移除 IOU, penalty 調整 TH = 0.5

## 20240925
- train267 epoch 200, FP penalty = 4000

## 20240929
- train278 epoch 1xx, 改 conf loss 但是太慢 (結果是 Dataset 拿錯)
- train286 epoch 6x, focal loss
- train288 epoch 59, focal loss with FN FP 權重
- train294 epoch 200, focal loss with FN FP 權重

## 20241002
- train302 epoch 200, focal loss with 最外層的 weight fb113b3

## 20241005
- train318 epoch 200, DFL + conf + dxdy: 0382e0d

## 20241006
- train322 epoch 200, DFL + conf + dxdy: eba7aca
- train328 epoch 200, DFL + conf

## 20241021
- train341 epoch 200, DFL + conf: 4eec5a6, val mode + save val (此 val 是 training 沒有看過的資料)

## 20241027
- val total ball count = 3904

## 20241027
- train348 epoch 200, conf+xy val 目前最佳, 可拿來做為之後比較依據 (ac1ec91fd13b772af3670146b9c33fbdb674d7d0)


## 20241101
- train354 epoch 200, conf+xy, 增加快速球的權重 dist = 20

12:50 包含 之前

## 20241104
- train371 60FPS + 120 FPS


## 20241110
- 距離相差2倍 hitV2- TP: 143, FP: 65, TN: 1806, FN: 236
- 距離相差1.5倍 

## 20241118
- train429 f205932a7638d524ab513846956c37d334264d20
  - 修好 fitness
  - 找到 loss 有負數的原因
  - precision-recall 圖，正確產出
- train431 35583f087ad425f97d86c025971839c6efd064a5
  - 修好 FP 的計算，IOU 太小也算預測有球但預測錯誤

## 20241123
- Ctrl + P 再加上 Ctrl + Q
  - 可以不中斷 python 執行
  - docker exec <container_name> ps aux => 可以檢查容器內執行的進程

- 執行 python script.py & => Ctrl + P 再加上 Ctrl + Q
  - 可以讓 python 在背景執行
- train437 b91725148efcca01f362cf3a55e7a975f7045c92
  - 不使用 conf weight
  - hit duplicate: 5

## 20241129
- train440 weight+hit duplicate: 5
  - 效果沒有比較好

## 20241201
*更換分支到 feat/detect-on-p3*
- train442
  - use p3
  - only weight (without hit duplicate)
- 單一 cell 只能偵測一個相同的物品
- w h 可以大於 cell 大小
  - .matmul(self.proj.type(pred_dist.dtype))
  - 這裡的 self.proj，決定的物品最大範圍 （離散的區間大小）

- train443 120, 60, 40 FPS (其餘同 442 程式碼)
  - f22ecbe6597421e23eab493e513caac84b4bb6e0

- blion: 1_05_03 271, 272 兩個 frame 相同

## 20241203
- TODO
- 修正 val loss

# 20241208
- train450
  - 改為中心點 anchor

- train451
  - add next x, y prediction
  - in the same head with current x, y
- train473
  - 修復 next xy
  - print next (x, y) when validate
- train475
  - adjust conf weight, more focus on FN
  - change focal loss hyper param (0.85, 1.5) 

# 20250120
- train502
  - 綜合可選的多樣 dataset 資料進行訓練
  - 尚未實作隨機旋轉與縮放
  - 關閉 nms (效能問題，待解決)

# 20250219
- training 點子
  - 將同一份 dataset 切成 n 份，拿第 i 份訓練結果，當作第 i+1 份訓練時的 penalty 權重依據

# 20250221
- train506 -- epoch 200 (9e2c28f6efca44aeedf098231c752068d1e2827e)
  - 引用 tracknetv3 Background Estimation
  - ```
    docker run --gpus all --ipc=host \
    -v /hdd/dataset/alex_tracknet:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/profession_match_2 \
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/profession_match_3 \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/profession_match_4 \
    -v /hdd/dataset/ces2025_all_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -it tracknetv4
    ```

-train517 (650c5fb7262aa2fa8a00c4303179d5364c18e20d)
  - fix 影像疊重疊問題
  - cache 不使用 docker 空間，使用 docker volume
  - fix val image tensor 沒有正規劃的情形 （應該是導致 val loss 異常的主因）
  - ```
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
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.cache:/usr/src/datasets/tracknet/train_data/.cache \
    -it tracknetv4
    ```
# 20250228
- train521 dd7bfef6bc55c3dd1095f4d7285da8524f376ec9
  - 可以繼續 train，數據感覺可以在往上

- train525 25b9a24817edd7727941104a49a21d65f7a97d2b
  - 每個 epoch 會重新調整 data sample 權重


- train527 ** => 87%
  - 接續 train525 epoch 7 (last.pt) 訓練 6adc681282c05afa7f6e414d334734ebee1f9fb3
  - 調整 conf weight
  - training 時，顯示 val image loss

- train534 69e3c4e19252f5fd04bd07264cd7ee49443b6d82
  - continue with train527 last | train530 last
  - process 處理發球羽球落地時，多餘的標記
  - 移除 weight pattern => 效果不佳
- train536 09c53e089e89365ef16b2dd86075052c6bde97d6
  - continue with train534
  - open OHEM
  - movement_threshold = 5
- train537 3218da330a5820875e29d18d3b29bbf2a2a3385c
  - continue with train536 
  - use log normalize on sampler
- train539 716e43f2918ec2c00c7cc814dc4046bf55159041
  - continue with train537
  - 降低 conf threshold = 0.5
  - sampler 分別 pos, conf loss 進行正規化
  - 最高到 89% (tol = 3 pixel)

------------------- 開始重新跑數據 -------------------

# 20250304
- train545 cf9d1ec7cce6c3abe9375958f6d951ec4b6767e5
  - only use background estimate 79%
  - ```
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
    -v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_13 \
    -v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_14 \
    -v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_15 \
    -v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/val_data/profession_match_21 \
    -v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/val_data/profession_match_22 \
    -v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/val_data/profession_match_23 \
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/val_data/profession_match_24 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/val_data/profession_match_25 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/val_data/profession_match_26 \
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4
    ```

- train548 3df3c6385ac112d5f86e04b8932aff814783e549
  - only use OHEM 77%
  - ```
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
    -v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_13 \
    -v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_14 \
    -v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_15 \
    -v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/val_data/profession_match_21 \
    -v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/val_data/profession_match_22 \
    -v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/val_data/profession_match_23 \
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/val_data/profession_match_24 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/val_data/profession_match_25 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/val_data/profession_match_26 \
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.n_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.n_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4
    ```

- train549 f36e47486c776d840345329aaa3edec7482e425f
  - only use down sample 80%
  - ```
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
    -v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_13 \
    -v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_14 \
    -v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_15 \
    -v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/val_data/profession_match_21 \
    -v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/val_data/profession_match_22 \
    -v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/val_data/profession_match_23 \
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/val_data/profession_match_24 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/val_data/profession_match_25 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/val_data/profession_match_26 \
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.n_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.n_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4
    ```

- train552 1510056bc3a9a651340de8e1ccd9b7096c9b7207 79%
  - dfl with 8 feature
  - 沒有混用其他方法
  - fix: 調整 dfl 換算 target 到連散數據的算法

- train555 0f031a98a6b00402f8301999abf0a36c0faf1c43
  - mse with position
  - 沒有混用其他方法

- train557 4d24410406a9e12e7f42c66ce53cad44c57c70a4
  - dfl with 2 feature

- train558 c117f0ba9208be910e8d9099949376e88affd4e1
  - smooth L1 with position

- train559 ac90cda84a4a83b27824c4778e36f1782f591797
  - smooth L1 with position and next position
  - four decouple head (x,y), conf, (nx,ny), n_conf
  - pure

- train560 dd315bc1415c5e6cd8f913bb46ce3f827cec7498
  - dfl only xy with 4 feature
  - two decouple head position, conf
  - pure

- train561 471086472ed6d1a2cd4997f094185ac2da3cde21
  - sl1 xy nxny in same head
  - pure

- train563 2184f315807001e61ed2ab0e64ac3fbc89b3616d
  - sl1 xy nxy in different head with different conf
  - adjust input label format: dxdy => nxny
  - pure
- train568 88e0807e2015eb6434c1933a36aa56478014e496
  - add dxdy loss on other head
  - pure

- train576 1a91dfcfef35db173036b390bf382ed5b8f1c856
  - train on p3 p4 p5, val on p3
  - dfl 8 feature
  - add n_conf feature
  - pure

- trainXXX bee31584a580275c4fb066c471176fa7b224a216
  - down FPS + background remove
  - dfl with 8 feature
  - use next target not dxdy
  - ```
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
    -v /hdd/dataset/profession_match_9:/usr/src/datasets/tracknet/train_data/profession_match_13 \
    -v /hdd/dataset/profession_match_10:/usr/src/datasets/tracknet/train_data/profession_match_14 \
    -v /hdd/dataset/profession_match_11:/usr/src/datasets/tracknet/train_data/profession_match_15 \
    -v /hdd/dataset/profession_match_12:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_13:/usr/src/datasets/tracknet/val_data/profession_match_21 \
    -v /hdd/dataset/profession_match_14:/usr/src/datasets/tracknet/val_data/profession_match_22 \
    -v /hdd/dataset/profession_match_15:/usr/src/datasets/tracknet/val_data/profession_match_23 \
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/val_data/profession_match_24 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/val_data/profession_match_25 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/val_data/profession_match_26 \
    -v /hdd/dataset/blion_tracknet_partial:/usr/src/datasets/tracknet/val_data/profession_match_20 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4
    ```


- train592 23f725cfaf029f1eb78887b27d4b17e839e4bda9
  - use p3p4p5 + one conf
  - 狀況異常，只有％10的圖片被偵測到 （第一針）
- train597 946ee199b5c4bd00f78c97e78face60a3acbc06d
  - use p3 + one conf
  - 修該為可支援 p3p4p5 架構
  - 測試該架構正常性(將 xy nxny 拆開算 loss) => 測試正常
- train598 3c67c314ea4da78427142003c5410b2f84b3133e
  - 同 train597 但開啟 p3p4p5
  - 正常可預測
- train599 b68a9921a8df44645daa704f79e88b68f7fa7961
  - use p3p4p5 + two conf
- train600 770438c1e53b069b42faa10fbc546ba86b7c7332
  - p3p4p5
  - use frame rate down sampling
- train603 bc9ebf0ea89668c4395bf9e497666a3b5dfcd81a
  - p345
  - frame rate down
  - background remove
- train604 continue on train603

- TODO
  - 移除 /hdd/dataset/profession_match_4 資料集
    - 有放大版的經典畫面，會混淆學習


## note
- 30FPS
  - speed_threshold = 15.0
    min_static_frames = 10
    smoothing_window = 1
    max_missing_frames = 20

- train608 3db2dcef88b6c8dc0d4a007cfd42db1e226c2450
  - ```
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
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_17:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_18:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_19:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/train_data/profession_match_21 \
    -v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/train_data/profession_match_22 \
    -v /hdd/dataset/profession_match_23:/usr/src/datasets/tracknet/train_data/profession_match_23 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_24 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_25 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_26 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_27 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/train_data/profession_match_28 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/train_data/profession_match_29 \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/profession_match_16_test:/usr/src/datasets/tracknet/val_data/profession_match_16_test \
    -v /hdd/dataset/profession_match_17_test:/usr/src/datasets/tracknet/val_data/profession_match_17_test \
    -v /hdd/dataset/profession_match_18_test:/usr/src/datasets/tracknet/val_data/profession_match_18_test \
    -v /hdd/dataset/profession_match_19_test:/usr/src/datasets/tracknet/val_data/profession_match_19_test \
    -v /hdd/dataset/profession_match_20_test:/usr/src/datasets/tracknet/val_data/profession_match_20_test \
    -v /hdd/dataset/profession_match_21_test:/usr/src/datasets/tracknet/val_data/profession_match_21_test \
    -v /hdd/dataset/profession_match_22_test:/usr/src/datasets/tracknet/val_data/profession_match_22_test \
    -v /hdd/dataset/profession_match_23_test:/usr/src/datasets/tracknet/val_data/profession_match_23_test \
    -v /hdd/dataset/profession_match_24_test:/usr/src/datasets/tracknet/val_data/profession_match_24_test \
    -v /hdd/dataset/profession_match_25_test:/usr/src/datasets/tracknet/val_data/profession_match_25_test \
    -v /hdd/dataset/profession_match_26_test:/usr/src/datasets/tracknet/val_data/profession_match_26_test \
    -v /hdd/dataset/profession_match_27_test:/usr/src/datasets/tracknet/val_data/profession_match_27_test \
    -v /hdd/dataset/profession_match_28_test:/usr/src/datasets/tracknet/val_data/profession_match_28_test \
    -v /hdd/dataset/profession_match_29_test:/usr/src/datasets/tracknet/val_data/profession_match_29_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.p_math_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.p_math_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4

    ```
  - p3p4p5
  - background removal
  - FPS downsampling
  - resample
  - refactor static ball algo
- train618 2bdcc405025453ec5731eb66b63dbe36094c23f0
  - 修改靜止球算法 V3
  - 使用所有方法
  - 調整 resample 的參數
  - 修正 nxny loss 計算時機（要這球＋下球都有球才會計算）


  - epoch 2
    - {'index': 9584, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/622.png'}    球落地途中被身體擋住，又出現
      {'index': 10228, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/624.png'}   球落地途中被身體擋住，又出現
      {'index': 41014, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_6/frame/1_05_03/103.png'}   靠近身體
      {'index': 10229, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/625.png'}   球落地途中被身體擋住，又出現
      {'index': 9586, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/624.png'}    球落地途中被身體擋住，又出現
      {'index': 41015, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_6/frame/1_05_03/104.png'}   靠近身體
      {'index': 17346, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_7/frame/2_05_03/900.png'}   觸網球
      {'index': 11608, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_16/frame/1_03_06/579.png'}  球落地旋轉（球頭沒有移動）
      {'index': 17345, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_7/frame/2_05_03/899.png'}   觸網球
      {'index': 12527, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_23/frame/1_06_04/334.png'}  多次中斷球看不見（被人擋著
  - epoch 3
    - {'index': 12435, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_23/frame/1_06_04/242.png'}    球頭與背景重疊 看不到
      {'index': 7055, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/941.png'}     球落地旋轉（球頭沒有移動）
      {'index': 7056, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/942.png'}     球落地旋轉（球頭沒有移動）
      {'index': 34113, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_29/frame/1_02_00/679.png'}    球落地 與網子重疊 被遮擋
      {'index': 30471, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_2/frame/1_00_02/18.png'}      無球 背景在動
      {'index': 2300, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_21/frame/1_02_01/344.png'}     球落地 沒什麼動
      {'index': 2301, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_21/frame/1_02_01/345.png'}     球落地 沒什麼動
      {'index': 34112, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_29/frame/1_02_00/678.png'}    球落地 與網子重疊 被遮擋
      {'index': 2299, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_21/frame/1_02_01/343.png'}     球落地 沒什麼動
      {'index': 16438, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_13/frame/1_09_10/530.png'}    觸網
  - epoch 6
    - {'index': 29118, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/305.png'}     畫面無球，特寫球員
      {'index': 33530, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_29/frame/1_02_00/96.png'}     擊球瞬間，球模糊
      {'index': 57099, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_11/frame/1_07_06/27.png'}     無球，發球前準備動作
      {'index': 53714, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_15/frame/2_14_08/164.png'}    球與地板線重疊，平行飛行
      {'index': 53711, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_15/frame/2_14_08/161.png'}    球與地板線重疊，平行飛行
      {'index': 53712, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_15/frame/2_14_08/162.png'}    球與地板線重疊，平行飛行
      {'index': 53713, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_15/frame/2_14_08/163.png'}    球與地板線重疊，平行飛行
      {'index': 29119, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/306.png'}     畫面無球，特寫球員
      {'index': 50548, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_17/frame/2_01_01/478.png'}    球落地途中被身體擋住，又出現
      {'index': 50026, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_17/frame/2_01_01/478.png'}    球落地途中被身體擋住，又出現
  - epoch 7
    - {'index': 9719, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/115.png'}
      {'index': 9080, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/118.png'}
      {'index': 32061, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_2/frame/1_06_09/414.png'}
      {'index': 9082, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/120.png'}
      {'index': 6724, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/610.png'}
      {'index': 29118, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/305.png'}
      {'index': 6722, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/608.png'}
      {'index': 29116, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/303.png'}
      {'index': 29117, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/304.png'}
      {'index': 9081, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/119.png'}
  - epoch 10
    - {'index': 44557, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_19/frame/1_01_01/428.png'}
      {'index': 40908, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_8/frame/2_03_06/1010.png'}
      {'index': 6195, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/81.png'}
      {'index': 15894, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_13/frame/1_09_10/533.png'}
      {'index': 16441, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_13/frame/1_09_10/533.png'}
      {'index': 44556, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_19/frame/1_01_01/427.png'}
      {'index': 29118, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/305.png'}
      {'index': 29117, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/304.png'}
      {'index': 29116, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/303.png'}
      {'index': 29119, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/306.png'}


profession_match_9/1_02_03 最後球落地的階段 有標記了非球頭的狀況

- train619
  - ```
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
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_17:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_18:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_19:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/train_data/profession_match_21 \
    -v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/train_data/profession_match_22 \
    -v /hdd/dataset/profession_match_23:/usr/src/datasets/tracknet/train_data/profession_match_23 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_24 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_25 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_26 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_27 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/train_data/profession_match_28 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/train_data/profession_match_29 \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/profession_match_16_test:/usr/src/datasets/tracknet/val_data/profession_match_16_test \
    -v /hdd/dataset/profession_match_17_test:/usr/src/datasets/tracknet/val_data/profession_match_17_test \
    -v /hdd/dataset/profession_match_18_test:/usr/src/datasets/tracknet/val_data/profession_match_18_test \
    -v /hdd/dataset/profession_match_19_test:/usr/src/datasets/tracknet/val_data/profession_match_19_test \
    -v /hdd/dataset/profession_match_20_test:/usr/src/datasets/tracknet/val_data/profession_match_20_test \
    -v /hdd/dataset/profession_match_21_test:/usr/src/datasets/tracknet/val_data/profession_match_21_test \
    -v /hdd/dataset/profession_match_22_test:/usr/src/datasets/tracknet/val_data/profession_match_22_test \
    -v /hdd/dataset/profession_match_23_test:/usr/src/datasets/tracknet/val_data/profession_match_23_test \
    -v /hdd/dataset/profession_match_24_test:/usr/src/datasets/tracknet/val_data/profession_match_24_test \
    -v /hdd/dataset/profession_match_25_test:/usr/src/datasets/tracknet/val_data/profession_match_25_test \
    -v /hdd/dataset/profession_match_26_test:/usr/src/datasets/tracknet/val_data/profession_match_26_test \
    -v /hdd/dataset/profession_match_27_test:/usr/src/datasets/tracknet/val_data/profession_match_27_test \
    -v /hdd/dataset/profession_match_28_test:/usr/src/datasets/tracknet/val_data/profession_match_28_test \
    -v /hdd/dataset/profession_match_29_test:/usr/src/datasets/tracknet/val_data/profession_match_29_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.mix_p_math_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.mix_p_math_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknetv4

    ```



    - ```
    docker run --gpus all --ipc=host \
    -v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.p_math_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.p_math_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -v /hdd/dataset/tracknetv4/profiler_output:/usr/src/ultralytics/profiler_output \
    -it tracknet1000 

    docker run --gpus all --ipc=host \
    -v /hdd/dataset/profession_match_1:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /hdd/dataset/sportxai_2025:/usr/src/datasets/tracknet/val_data/sportxai_2025 \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.p_math_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.p_math_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -v /hdd/dataset/tracknetv4/profiler_output:/usr/src/ultralytics/profiler_output \
    -it tracknet1000 

    docker run --gpus all --ipc=host \
    -v /home/bartektao/dataset/sportxai_2025:/usr/src/datasets/tracknet/train_data/profession_match_1 \
    -v /home/bartektao/dataset/sportxai_2025:/usr/src/datasets/tracknet/val_data/sportxai_2025 \
    -v /home/bartektao/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /home/bartektao/dataset/tracknetv4/profiler_output:/usr/src/ultralytics/profiler_output \
    -it tracknet1000 

    c

    python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train637/weights/best.pt --source /usr/src/datasets/tracknet/val_data/sport_ai_2048_1536/frame/CameraReader_1/

    python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 20

    python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train618/weights/best.pt --source /usr/src/datasets/tracknet/val_data/profession_match_15_test/frame/2_18_14/

    python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train618/weights/best.pt --source /usr/src/datasets/tracknet/val_data/sportxai_2025/frame/2025-01-16_15-18-59_0/

    python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train637/weights/best.pt --source /usr/src/datasets/tracknet/val_data/profession_match_15_test/frame/2_18_14/

    python tracknet.py --mode predict_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train637/weights/best.pt --source /usr/src/datasets/tracknet/val_data/sportxai_2025/frame/2025-01-16_15-18-59_0/

    python tracknet.py --mode val_v2 --batch 1 --model_path /usr/src/ultralytics/runs/detect/train618/weights/best.pt --source /usr/src/datasets/tracknet/val_data

    export CUDA_LAUNCH_BLOCKING=1
    unset CUDA_LAUNCH_BLOCKING

    watch -n 1 nvidia-smi

    pip install tensorboard

    tensorboard --logdir=./profiler_output

    ```

{
  "calibration": {
    "near_camera_head_width_px": 20.0,
  }
}

p_match
  15*1000 = 15000
AUX_nycu_new_court
  2000
nycu_new_court_2048_1536(無 test)
  2000
sportxai_serve_machine
  2000
sportxai_rally
  2000
hsinchu_gym
  2000

ces2025_all
  2000
office_dataset
  2000

- train627
  - ```
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
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/AUX_nycu_new_court \
    -v /hdd/dataset/nycu_new_court_2048_1536:/usr/src/datasets/tracknet/train_data/nycu_new_court_2048_1536 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/sportxai_serve_machine \
    -v /hdd/dataset/sportxai_rally:/usr/src/datasets/tracknet/train_data/sportxai_rally \
    -v /hdd/dataset/hsinchu_gym:/usr/src/datasets/tracknet/train_data/hsinchu_gym \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/ces2025_all \
    -v /hdd/dataset/office_dataset:/usr/src/datasets/tracknet/train_data/office_dataset \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/AUX_nycu_new_court_test:/usr/src/datasets/tracknet/train_data/AUX_nycu_new_court_test \
    -v /hdd/dataset/sportxai_serve_machine_test:/usr/src/datasets/tracknet/train_data/sportxai_serve_machine_test \
    -v /hdd/dataset/sportxai_rally_test:/usr/src/datasets/tracknet/train_data/sportxai_rally_test \
    -v /hdd/dataset/hsinchu_gym_test:/usr/src/datasets/tracknet/train_data/hsinchu_gym_test \
    -v /hdd/dataset/ces2025_all_test:/usr/src/datasets/tracknet/train_data/ces2025_all_test \
    -v /hdd/dataset/office_dataset_test:/usr/src/datasets/tracknet/train_data/office_dataset_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.br_all_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.br_all_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknet1000

    python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/ultralytics/models/v8/tracknetv4.yaml --epoch 20 &

    docker ID 4012ae496663

    ```

- train628 2563d730c5c2a6041b30fdd2aff40ed252fb94d3
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
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_17:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_18:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_19:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/train_data/profession_match_21 \
    -v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/train_data/profession_match_22 \
    -v /hdd/dataset/profession_match_23:/usr/src/datasets/tracknet/train_data/profession_match_23 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_24 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_25 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_26 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_27 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/train_data/profession_match_28 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/train_data/profession_match_29 \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/profession_match_16_test:/usr/src/datasets/tracknet/val_data/profession_match_16_test \
    -v /hdd/dataset/profession_match_17_test:/usr/src/datasets/tracknet/val_data/profession_match_17_test \
    -v /hdd/dataset/profession_match_18_test:/usr/src/datasets/tracknet/val_data/profession_match_18_test \
    -v /hdd/dataset/profession_match_19_test:/usr/src/datasets/tracknet/val_data/profession_match_19_test \
    -v /hdd/dataset/profession_match_20_test:/usr/src/datasets/tracknet/val_data/profession_match_20_test \
    -v /hdd/dataset/profession_match_21_test:/usr/src/datasets/tracknet/val_data/profession_match_21_test \
    -v /hdd/dataset/profession_match_22_test:/usr/src/datasets/tracknet/val_data/profession_match_22_test \
    -v /hdd/dataset/profession_match_23_test:/usr/src/datasets/tracknet/val_data/profession_match_23_test \
    -v /hdd/dataset/profession_match_24_test:/usr/src/datasets/tracknet/val_data/profession_match_24_test \
    -v /hdd/dataset/profession_match_25_test:/usr/src/datasets/tracknet/val_data/profession_match_25_test \
    -v /hdd/dataset/profession_match_26_test:/usr/src/datasets/tracknet/val_data/profession_match_26_test \
    -v /hdd/dataset/profession_match_27_test:/usr/src/datasets/tracknet/val_data/profession_match_27_test \
    -v /hdd/dataset/profession_match_28_test:/usr/src/datasets/tracknet/val_data/profession_match_28_test \
    -v /hdd/dataset/profession_match_29_test:/usr/src/datasets/tracknet/val_data/profession_match_29_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.br_all_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.br_all_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknet1000

- train631 base on train630 2563d730c5c2a6041b30fdd2aff40ed252fb94d3
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
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_17:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_18:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_19:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/train_data/profession_match_21 \
    -v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/train_data/profession_match_22 \
    -v /hdd/dataset/profession_match_23:/usr/src/datasets/tracknet/train_data/profession_match_23 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_24 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_25 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_26 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_27 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/train_data/profession_match_28 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/train_data/profession_match_29 \
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/AUX_nycu_new_court \
    -v /hdd/dataset/nycu_new_court_2048_1536:/usr/src/datasets/tracknet/train_data/nycu_new_court_2048_1536 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/sportxai_serve_machine \
    -v /hdd/dataset/sportxai_rally:/usr/src/datasets/tracknet/train_data/sportxai_rally \
    -v /hdd/dataset/hsinchu_gym:/usr/src/datasets/tracknet/train_data/hsinchu_gym \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/ces2025_all \
    -v /hdd/dataset/office_dataset:/usr/src/datasets/tracknet/train_data/office_dataset \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/profession_match_16_test:/usr/src/datasets/tracknet/val_data/profession_match_16_test \
    -v /hdd/dataset/profession_match_17_test:/usr/src/datasets/tracknet/val_data/profession_match_17_test \
    -v /hdd/dataset/profession_match_18_test:/usr/src/datasets/tracknet/val_data/profession_match_18_test \
    -v /hdd/dataset/profession_match_19_test:/usr/src/datasets/tracknet/val_data/profession_match_19_test \
    -v /hdd/dataset/profession_match_20_test:/usr/src/datasets/tracknet/val_data/profession_match_20_test \
    -v /hdd/dataset/profession_match_21_test:/usr/src/datasets/tracknet/val_data/profession_match_21_test \
    -v /hdd/dataset/profession_match_22_test:/usr/src/datasets/tracknet/val_data/profession_match_22_test \
    -v /hdd/dataset/profession_match_23_test:/usr/src/datasets/tracknet/val_data/profession_match_23_test \
    -v /hdd/dataset/profession_match_24_test:/usr/src/datasets/tracknet/val_data/profession_match_24_test \
    -v /hdd/dataset/profession_match_25_test:/usr/src/datasets/tracknet/val_data/profession_match_25_test \
    -v /hdd/dataset/profession_match_26_test:/usr/src/datasets/tracknet/val_data/profession_match_26_test \
    -v /hdd/dataset/profession_match_27_test:/usr/src/datasets/tracknet/val_data/profession_match_27_test \
    -v /hdd/dataset/profession_match_28_test:/usr/src/datasets/tracknet/val_data/profession_match_28_test \
    -v /hdd/dataset/profession_match_29_test:/usr/src/datasets/tracknet/val_data/profession_match_29_test \
    -v /hdd/dataset/sportxai_serve_machine_test:/usr/src/datasets/tracknet/val_data/sportxai_serve_machine_test \
    -v /hdd/dataset/sportxai_rally_test:/usr/src/datasets/tracknet/val_data/sportxai_rally_test \
    -v /hdd/dataset/hsinchu_gym_test:/usr/src/datasets/tracknet/val_data/hsinchu_gym_test \
    -v /hdd/dataset/ces2025_all_test:/usr/src/datasets/tracknet/val_data/ces2025_all_test \
    -v /hdd/dataset/office_dataset_test:/usr/src/datasets/tracknet/val_data/office_dataset_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.br_all_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.br_all_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknet1000

    python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train630/weights/last.pt --epoch 20 &

- train632 base on train631 2563d730c5c2a6041b30fdd2aff40ed252fb94d3
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
    -v /hdd/dataset/profession_match_16:/usr/src/datasets/tracknet/train_data/profession_match_16 \
    -v /hdd/dataset/profession_match_17:/usr/src/datasets/tracknet/train_data/profession_match_17 \
    -v /hdd/dataset/profession_match_18:/usr/src/datasets/tracknet/train_data/profession_match_18 \
    -v /hdd/dataset/profession_match_19:/usr/src/datasets/tracknet/train_data/profession_match_19 \
    -v /hdd/dataset/profession_match_20:/usr/src/datasets/tracknet/train_data/profession_match_20 \
    -v /hdd/dataset/profession_match_21:/usr/src/datasets/tracknet/train_data/profession_match_21 \
    -v /hdd/dataset/profession_match_22:/usr/src/datasets/tracknet/train_data/profession_match_22 \
    -v /hdd/dataset/profession_match_23:/usr/src/datasets/tracknet/train_data/profession_match_23 \
    -v /hdd/dataset/profession_match_24:/usr/src/datasets/tracknet/train_data/profession_match_24 \
    -v /hdd/dataset/profession_match_25:/usr/src/datasets/tracknet/train_data/profession_match_25 \
    -v /hdd/dataset/profession_match_26:/usr/src/datasets/tracknet/train_data/profession_match_26 \
    -v /hdd/dataset/profession_match_27:/usr/src/datasets/tracknet/train_data/profession_match_27 \
    -v /hdd/dataset/profession_match_28:/usr/src/datasets/tracknet/train_data/profession_match_28 \
    -v /hdd/dataset/profession_match_29:/usr/src/datasets/tracknet/train_data/profession_match_29 \
    -v /hdd/dataset/AUX_nycu_new_court:/usr/src/datasets/tracknet/train_data/AUX_nycu_new_court \
    -v /hdd/dataset/nycu_new_court_2048_1536:/usr/src/datasets/tracknet/train_data/nycu_new_court_2048_1536 \
    -v /hdd/dataset/sportxai_serve_machine:/usr/src/datasets/tracknet/train_data/sportxai_serve_machine \
    -v /hdd/dataset/sportxai_rally:/usr/src/datasets/tracknet/train_data/sportxai_rally \
    -v /hdd/dataset/hsinchu_gym:/usr/src/datasets/tracknet/train_data/hsinchu_gym \
    -v /hdd/dataset/ces2025_all:/usr/src/datasets/tracknet/train_data/ces2025_all \
    -v /hdd/dataset/office_dataset:/usr/src/datasets/tracknet/train_data/office_dataset \
    -v /hdd/dataset/profession_match_1_test:/usr/src/datasets/tracknet/val_data/profession_match_1_test \
    -v /hdd/dataset/profession_match_2_test:/usr/src/datasets/tracknet/val_data/profession_match_2_test \
    -v /hdd/dataset/profession_match_3_test:/usr/src/datasets/tracknet/val_data/profession_match_3_test \
    -v /hdd/dataset/profession_match_4_test:/usr/src/datasets/tracknet/val_data/profession_match_4_test \
    -v /hdd/dataset/profession_match_5_test:/usr/src/datasets/tracknet/val_data/profession_match_5_test \
    -v /hdd/dataset/profession_match_6_test:/usr/src/datasets/tracknet/val_data/profession_match_6_test \
    -v /hdd/dataset/profession_match_7_test:/usr/src/datasets/tracknet/val_data/profession_match_7_test \
    -v /hdd/dataset/profession_match_8_test:/usr/src/datasets/tracknet/val_data/profession_match_8_test \
    -v /hdd/dataset/profession_match_9_test:/usr/src/datasets/tracknet/val_data/profession_match_9_test \
    -v /hdd/dataset/profession_match_10_test:/usr/src/datasets/tracknet/val_data/profession_match_10_test \
    -v /hdd/dataset/profession_match_11_test:/usr/src/datasets/tracknet/val_data/profession_match_11_test \
    -v /hdd/dataset/profession_match_12_test:/usr/src/datasets/tracknet/val_data/profession_match_12_test \
    -v /hdd/dataset/profession_match_13_test:/usr/src/datasets/tracknet/val_data/profession_match_13_test \
    -v /hdd/dataset/profession_match_14_test:/usr/src/datasets/tracknet/val_data/profession_match_14_test \
    -v /hdd/dataset/profession_match_15_test:/usr/src/datasets/tracknet/val_data/profession_match_15_test \
    -v /hdd/dataset/profession_match_16_test:/usr/src/datasets/tracknet/val_data/profession_match_16_test \
    -v /hdd/dataset/profession_match_17_test:/usr/src/datasets/tracknet/val_data/profession_match_17_test \
    -v /hdd/dataset/profession_match_18_test:/usr/src/datasets/tracknet/val_data/profession_match_18_test \
    -v /hdd/dataset/profession_match_19_test:/usr/src/datasets/tracknet/val_data/profession_match_19_test \
    -v /hdd/dataset/profession_match_20_test:/usr/src/datasets/tracknet/val_data/profession_match_20_test \
    -v /hdd/dataset/profession_match_21_test:/usr/src/datasets/tracknet/val_data/profession_match_21_test \
    -v /hdd/dataset/profession_match_22_test:/usr/src/datasets/tracknet/val_data/profession_match_22_test \
    -v /hdd/dataset/profession_match_23_test:/usr/src/datasets/tracknet/val_data/profession_match_23_test \
    -v /hdd/dataset/profession_match_24_test:/usr/src/datasets/tracknet/val_data/profession_match_24_test \
    -v /hdd/dataset/profession_match_25_test:/usr/src/datasets/tracknet/val_data/profession_match_25_test \
    -v /hdd/dataset/profession_match_26_test:/usr/src/datasets/tracknet/val_data/profession_match_26_test \
    -v /hdd/dataset/profession_match_27_test:/usr/src/datasets/tracknet/val_data/profession_match_27_test \
    -v /hdd/dataset/profession_match_28_test:/usr/src/datasets/tracknet/val_data/profession_match_28_test \
    -v /hdd/dataset/profession_match_29_test:/usr/src/datasets/tracknet/val_data/profession_match_29_test \
    -v /hdd/dataset/sportxai_serve_machine_test:/usr/src/datasets/tracknet/val_data/sportxai_serve_machine_test \
    -v /hdd/dataset/sportxai_rally_test:/usr/src/datasets/tracknet/val_data/sportxai_rally_test \
    -v /hdd/dataset/hsinchu_gym_test:/usr/src/datasets/tracknet/val_data/hsinchu_gym_test \
    -v /hdd/dataset/ces2025_all_test:/usr/src/datasets/tracknet/val_data/ces2025_all_test \
    -v /hdd/dataset/office_dataset_test:/usr/src/datasets/tracknet/val_data/office_dataset_test \
    -v /hdd/dataset/tracknetv4/runs:/usr/src/ultralytics/runs \
    -v /hdd/dataset/tracknetv4/visualize_train_img:/usr/src/datasets/tracknet/visualize_train_img \
    -v /hdd/dataset/tracknetv4/visualize_predict_img:/usr/src/datasets/tracknet/visualize_predict_img \
    -v /hdd/dataset/tracknetv4/val_confusion_matrix:/usr/src/datasets/tracknet/val_confusion_matrix \
    -v /hdd/dataset/tracknetv4/.br_all_cache:/usr/src/datasets/tracknet/train_data/.cache \
    -v /hdd/dataset/tracknetv4/.br_all_val_cache:/usr/src/datasets/tracknet/val_data/.cache \
    -it tracknet1000

    python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train631/weights/last.pt --epoch 20 &

    {'index': 204854, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/388.png'}
    {'index': 204855, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_4/frame/2_05_07/389.png'}
    {'index': 206008, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_2/frame/1_00_02/579.png'}
    {'index': 52830, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_10/frame/1_12_16/303.png'}
    {'index': 54962, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_22/frame/1_02_01/14.png'}
    {'index': 56413, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_1/frame/1_01_00/31.png'}
    {'index': 292228, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_19/frame/2_12_06/521.png'}
    {'index': 59417, 'img_files': '/usr/src/datasets/tracknet/train_data/profession_match_23/frame/1_11_08/17.png'}
    {'index': 217544, 'img_files': '/usr/src/datasets/tracknet/train_data/ces2025_all/frame/2025-01-07_06-56-40_0/77.png'} 
      label 異常
      77,1,600.0,456.0,0.0,0,3.4993928571428574,0
    {'index': 285587, 'img_files': '/usr/src/datasets/tracknet/train_data/sportxai_serve_machine/frame/2024-09-19_10-25-54_1/0.png'} 
      label 異常 與 heatmap 相同
      Frame,Visibility,X,Y,Z,Event,Timestamp,Fast
      0,0,1783.0,327.0,0.0,0,0.0,0
- train637
  - python tracknet.py --mode train_v2 --model_path /usr/src/ultralytics/runs/detect/train632/weights/last.pt --epoch 20 &


## 多球
/hdd/dataset/sport_ai_2048_1536/video/