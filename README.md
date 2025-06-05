# ML Assignment 4 - Challenges in Representation Learning, Facial Expression Recognition Challenge

კონკურსში ჩვენი მთავარი მიზანია გავწვრთნათ ნეირონული ქსელი, რომელიც ამოიცნობს ემოციებს ადამიანის სახეზე. Training Set-ის სახით მოცემულია 28, 709 სურათი და თითოეულს აქვს label (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). საბოლოოდ ჩვენი მოდელი შეფასდება accuracy-ით test set-ზე.

### ჩემი მიდგომა პრობლემის გადასაჭრელად

ამ ამოცანის გადასაჭრელად ვიყენებ Fully Connected Neural Nets(FCNN) და Convulutional Neural Networks(CNN)-ის კომბინაციას. თავიდან ვიწყებ უმარტივესი არქიტექტურით და მხოლოდ FCNN-ის გამოყებეით. მაინტერესებს, რამდენად კარგად დაისწავლის მონაცემებს მაგალითად 1-ფენიანი, 2-ფენიანი ან 3-ფენიანი FCNN, რა იქნება მომენტი, როდესაც Overfit შესამჩნევი გახდება, შემდეგ რა რეგულარიზაცია უნდა დავამატო, რომ Overfit შევამცირო და რამდენად გაუმჯობესდება performance თუ CNN layer-ებსაც ჩავრთავ. ჩემი მოდელები იზრდება იტერაციულად და მხოლოდ 1 ცვლილებით წინა მოდელებისგან განსხვავებით, რათა უკეთ გავაკეთო დასკვნები ჰიპერ-პარამეტრების შესახებ. თავიდან ყველა მოდელს გავწვრთნი 5 ეპოქის განმავლობაში და შემდეგ ამოვარჩევ საუკეთესოებს ამ სიიდან და ხელახლა გავწვრთნი უფრო დიდხანს.

# რეპოზიტორიის სტრუქტურა

- FCNN_CNN_Training_and_Inference.ipynb - Data loading, Preprocessing, Training და Inference ნაბიჯები

- README.md - მოდელების გაწვრთნის პროცესის დეტალური აღწერა


# Preprocessing

თავდაპირველი training set გავჭერი 70% - 15% - 15% ნაწილებად. პირველი ნაწილი გამოვიყენე training-ისთვის დანარჩენები კი validation და test set-ებად.

მოდელისათვის საქმის გასამარტივებლად გადავწყვიტე, რომ მონაცემების საშუალო გამეხადა 0-ის ტოლი. ამის გასაკეთებლად training set-დან გამოვთვალე საშუალო სურათი და ყველა set-ის ყველა სურათს გამოვაკელი ეს საშუალო სურათი.

data leakage რომ არ მიმეღო validation set-საც და test set-საც გამოვაკელი training-დან დაგენერირებული საშუალო სურათი.

ეს ყველა ნაბიჯი აღწერილია დეტალურად თავად `.ipynb` ფაილში.

# Training
გამოვიყენე FCNN და CNN არქიტექტურები მოდელებში და თითოეული არქიტექტურისათვის wandb-ზე შევქმენი პროექტი, სადაც სხვადასხვა ექსპერიმენტები ცალკე run-ებად ჩავყარე.

`.ipynb` ფაილში მოცემულია ქვემოთ ჩამოთვლილი ყველა მოდელის კოდი მთლიანად, იმავე სათაურების ქვეშ.

## Fully Connected Neural Networks

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork


### Experiment 1: Linear-FCN-One-Layer
https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/8nihvr71

თავდაპირველი ექსპერიმენტისათვის ავიღე უმარტივესი 1 ფენიანი წრფივი მოდელი. layer-ში ნეირონების რაოდენობა ავარჩიე ისე, რომ ყოფილიყო დაახლოებით 2-ჯერ მეტი სურათში პიქსელების რაოდენობაზე. Loss ფუნქციად გამოვიყენე `CrossEntropyLoss` და optimizer-ად სტანდარტული `SGD` ალგორითმი. ეს არის ყველაზე კლასიკური და უმარტივესი სტრუქტურა, საიდანაც შემიძლია, რომ მუშაობა დავიწყო.

როგორც ადვილი სავარაუდებელია, წრფივი მოდელი იყო არასაკმარისად კომპლექსური მონაცემებისათვის და შესაბამისად ჰქონდა მაღალი bias.

```
Correct: 5616 / 20096, Train Accuracy: 0.27945859872611467
Validation: 892 / 4306, Validation Accuracy: 0.20715281003251276
```

მიღებული შედეგები უფრო უკეთესია, ვიდრე პასუხის random შერჩევა, რაც 14% შემთხვევაში მოგვცემდა სწორ პასუხს. საჭიროა მოდელის კომპლექსურობის გაზრდა.

### Experiment 2: FCN-Two-Layer
https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/zvxryz2e

ორფენიანი Fully Connected Net უკვე საშუალებას გვაძლევს, რომ `ReLU`-ს სახით შემოვიტანოთ `non-linearity`. ამ მოდელში წინა მოდელისაგან განსხვავებით, მხოლოდ ის შევცვალე, რომ შემოვიტანე დამატებითი 4096 ნეირონიანი ფენა და ორ ფენას შორის ჩავსვი `ReLU`. შემოტანილმა კომპლექსურობამ შედარებით გააუმჯობესა შედეგები:

```
Correct: 8640 / 20096, Train Accuracy: 0.4299363057324841
Validation: 1361 / 4306, Validation Accuracy: 0.31607059916395724
```

### Experiment 3: FCN-Three-Layer-SGD

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/fksolbdu

შემდეგი ლოგიკური ნაბიჯია, რომ შევეცადო კიდევ გავზარდო ფენების რაოდენობა და დავაკვირდე გაუმჯობესდება თუ არა შედეგი. ჩემი ნეირონული ქსელი გავხადე ახლა 3-ფენიანი.

```
Correct: 19317 / 20096, Train Accuracy: 0.9612360668789809
Validation: 1584 / 4306, Validation Accuracy: 0.36785880167208546
```

თუმცა გამოვიდა, რომ ამ პატარა ცვლილებამ უკვე მოდელი გახადა overfitted. ეს იქიდან შეიმჩნევა, რომ training set-ზე accuracy გაცილებით უფრო დიდია, ვიდრე validation set-ზე.

### Experiment 4: FCN-Three-Layer-Dropout

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/pngrc0wu

წინა ექსპერიმენტმა გვაჩვენა, რომ FCNN-ის overfit ძალიან მარტივია რეგულაციის გარეშე. ამიტომაც ამ ექსპერიმენტში გადავწყვიტე დამემატებინა რეგულატორი `Dropout`. თავდაპირველად დასადროფი ნეირონების რაოდენობა ავიღე `dropout=0.1` და ამ ჰიპერ-პარამეტრს ექსპერიმენტებს შორის შევცვლი.

```
Correct: 11086 / 20096, Train Accuracy: 0.5516520700636943
Validation: 1459 / 4306, Validation Accuracy: 0.3388295401764979
```

overfit უფრო შემცირდა ამ შედეგებიდან გამომდინარე, თუმცა უფრო გაუმჯობესებაცაა შესაძლებელი dropout ჰიპერ-პარამეტრის გაზრდით.

### Experiment 5: FCN-Three-Layer-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/1xrtz44w

ამ ექსპერიმენტში `dropout=0.5`

```
Correct: 4487 / 20096, Train Accuracy: 0.2232782643312102
Validation: 1080 / 4306, Validation Accuracy: 0.25081281932187643
```

ამ შედეგებიდან ირკვევა, რომ dropout საკმაოდ ძლიერი რეგულარიზაციის ტექნიკაა. მართალია, ვალიდაციის ქულაც შემცირდა, თუმცა ეს პრობლემა გამოსწორდება თუ უფრო მეტ ეპოქაზე გავწვრთნით ამ მოდელს.

ასევე საინტერესოა, რომ ვალიდაციის ქულა უფრო უკეთესია, ვიდრე training-ის. ვფიქრობ, ეს იმიტომ ხდება, რომ forward pass-ის დროს ვალიდაციაზე სიგნალების dropout აღარ ხდება, ხოლო training-ის დროს ამ შემთხვევაში ნახევარი მოდელი გათიშულია.


### Experiment 6: FCN-Four-Layer-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/nd1su6ri

### Experiment 7: FCN-Five-Layer-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/75yc79hs

ამ ორ ექსპერიმენტს განვიხილავ ერთად. Dropout რეგულარიზაცია იწვევს იმას, რომ ორივე შემთხვევაში საგრძნობლად მცირდება validation და training accuracy და აღმოჩნდა, რომ ზემოთა სამივე ექსპერიმენტში 5 ეპოქის გაწვრთნის შემდეგ 20%-ია დაახლოებით ვალიდაციის ქულა. თეორიულად შესაძლებელია, რომ ეს უფრო deep ქსელები უფრო მეტ ეპოქაზე გავწვრთნა, თუმცა 5 ეპოქაზზე გაწვრთნამაც საკმაოდ დიდი დრო წაიღო და ჯერ-ჯერობით მირჩევნია, რომ 3 ფენიან ქსელებზე ვიმუშაო.


### Experiment 8: FCN-Three-Layer-Momentum

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/bnw2y9dh

როგორც ზემოთ ვთქვი, აქამდე ოპტიმიზატორად ვიყენებდი სტანდარტულ SGD-ის. აღმოჩნდა, რომ convergence ყველაზე სწრაფად არ ხდება SGD-ის შემთხვევაში და გადავწყვიტე Google Colab-ის GPU-ზე დამენიშნა რა დრო დასჭირდებოდა თითოეულ optimizer ალგორითმს და რა ფიქსირებული ეპოქების რაოდენობაში, რომელი მიაღწევდა ყველაზე უკეთეს შედეგს.

თავდაპირველად გავტესტე Momentum.

```
Correct: 10742 / 20096, Train Accuracy: 0.5345342356687898
Validation: 1346 / 4306, Validation Accuracy: 0.31258708778448674
Epoch training time: 190.6004753112793 seconds
-------------------------------------------

Total training time: 945.4743525981903 seconds
```

### Experiment 9: FCN-Three-Layer-Momentum-Nesterov

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/q9x60jy3

```
Correct: 11996 / 20096, Train Accuracy: 0.5969347133757962
Validation: 1383 / 4306, Validation Accuracy: 0.3211797491871807
Epoch training time: 204.3428087234497 seconds
-------------------------------------------

Total training time: 919.2456274032593 seconds
```

### Experiment 10: FCN-Three-Layer-RMSprop

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/j9za6ibk


```
Correct: 7434 / 20096, Train Accuracy: 0.36992436305732485
Validation: 1455 / 4306, Validation Accuracy: 0.3379006038086391
Epoch training time: 197.40231776237488 seconds
-------------------------------------------

Total training time: 979.5215811729431 seconds
```

### Experiment 11: FCN-Three-Layer-Adam

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/006zqxuu


```
Correct: 8068 / 20096, Train Accuracy: 0.40147292993630573
Validation: 1527 / 4306, Validation Accuracy: 0.3546214584300975
Epoch training time: 172.69184064865112 seconds
-------------------------------------------

Total training time: 842.0318641662598 seconds
```

ზემოთ მოყვანილი ექსპერიმენტებიდან გამომდინარე აღმოჩნდა, რომ Adam optimizer-მა საუკეთესო შედეგი მოგვცა ყველაზე ცოტა დროში. ამ შედეგებიდან გამომდინარე გადავწყვიტე, რომ შემდეგ მოდელებში SGD-ის ნაცვლად გამომეყენებინა Adam, რადგან უფრო კომპლექსურ მოდელებთან მექნება შეხება.

### Experiment 12: FCN-Three-Layer-BatchNorm

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/9jiowph7

შემდეგი ექსპერიმენტისათვის გადავწყვიტე, გამომეყენებინა Batch Normalization, რომელიც როგორც weight-ების საშუალოსა და ვარიაციას ცვლის, ასევე მოქმედებს რეგულარიზაციის საშუალებადაც. თავდაპირველი ექსპერიმენტი გავუშვი `batch_size=10` ჰიპერპარამეტრით.

```
Correct: 8059 / 20096, Train Accuracy: 0.4010250796178344
Validation: 1619 / 4306, Validation Accuracy: 0.37598699489084997
Epoch training time: 182.99799871444702 seconds
```

5 ეპოქაში მიღებული შედეგები იყო გაუმჯობესება, თუმცა აღსანიშნავია, რომ `batch_size=10`-ის გამო BatchNorm-ის მიერ გამოთვლილი ვარიაცია და საშუალო კარგად არ ასახავდა მთლიანი მონაცემების ამ პარამეტრებს.

### Experiment 13: FCN-Three-Layer-BatchNorm-BatchSize-100

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/3k1skz0f


```
Correct: 15178 / 20096, Train Accuracy: 0.7552746815286624
Validation: 1861 / 4306, Validation Accuracy: 0.4321876451463075
Epoch training time: 20.00678253173828 seconds
-------------------------------------------

Total training time: 95.68968963623047 seconds
```

`batch_size=100` დავსეტე ჰიპერ-პარამეტრად

batch_size-ის გაზრდამ გააუმჯობესა ვალიდაციის შედეგი და მოდელის გაწვრთნის დროს საგრძნობლად შეამცირა.


### Experiment 14: FCN-Three-Layer-BatchNorm-BatchSize-200

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/no7p32re

```
Correct: 16127 / 20096, Train Accuracy: 0.8024980095541401
Validation: 1842 / 4306, Validation Accuracy: 0.4277751973989782
Epoch training time: 9.505066871643066 seconds
-------------------------------------------

Total training time: 46.45437836647034 seconds
```

`batch_size=200`-ზე დასეტვა შედეგს დიდად არ ცვლის, რადგან აღმოჩნდა, რომ ამ შემთხვევაში 100 მაგალითიც საკმაოდ კარგად აღწერს საშუალოსა და ვარიაციას მთლიანი მონაცემების, თუმცა ამჯერად გამოვიყენებ მომავალ მოდელებში ამ პარამეტრს, რადგან training time-ს საგრძნობლად ამცირებს.

### Experiment 15: FCN-Five-Layer-BatchNorm

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/o0ub4or4


```
Correct: 19075 / 20096, Train Accuracy: 0.9491938694267515
Validation: 1829 / 4306, Validation Accuracy: 0.42475615420343704
Epoch training time: 20.232117176055908 seconds
-------------------------------------------

Total training time: 108.44212675094604 seconds
```

შემდეგ ექსპერიმენტში გადავწყვიტე, რომ მოდელის კომპლექსურობა კიდევ გამეზარდა 5 ფენიან FCNN-მდე, თუმცა ამჯერად BatchNorm-ითა და `batch_size=200`-ით. შედეგი სახეზეა, რომ მივიღე ისევ მკვეთრად overfitted მოდელი 5 ეპოქაში. აქედან შემიძლია დავასკვნა, რომ BatchNorm უფრო სუსტი რეგულარიზატორია, ვიდრე Dropout.

### Experiment 16: FCN-Five-Layer-BatchNorm-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/m4lpijsv

წინა ექსპერიმენტებიდან მიღებული შედეგები მინდა გავაერთიანო ერთ ექსპერიმენტში და ერთად გამოვიყენო BatchNorm და Dropout

`dropout=0.5` ავირჩიე პარამეტრად ამ ექსპერიმენტისათვის. ქსელი არის წინა ექსპერიმენტიდან 5-layer FCNN.


```
Correct: 5954 / 20096, Train Accuracy: 0.2962778662420382
Validation: 1585 / 4306, Validation Accuracy: 0.36809103576405017
Epoch training time: 22.537006616592407 seconds
-------------------------------------------

Total training time: 109.13190865516663 seconds
```

მივიღე Dropout-ისთვის დამახასიათებელი შედეგი, როდესაც `validation score > train score`

აღსანიშნავია, რომ BatchNorm-თან ერთად ვალიდაციის ქულა არის 36% 5 layer ქსელზე, როდესაც მხოლოდ Dropout-ით 5 layer network-ზე ეს ქულა იყო დაახლოებით 20% 5 ეპოქის შემდეგ.

### Experiment 17: FCN-Five-Layer-BatchNorm-Dropout-30

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-FullyConnectedNetwork/runs/igawfgzs

ამ შემდეგ ექსპერიმენტში გადავწყვიტე, რომ `dropout=0.3` დამესეტა და დავკვირვებოდი შედეგებს.


```
Correct: 7910 / 20096, Train Accuracy: 0.39361066878980894
Validation: 1725 / 4306, Validation Accuracy: 0.4006038086391082
Epoch training time: 20.3179452419281 seconds
-------------------------------------------

Total training time: 108.36702013015747 seconds
```

მივიღე ვალიდაციაზე 40%, რაც არის საუკეთესო შედეგი FCNN-ზე რაც კი მიმიღია, როდესაც train score დაახლოებით იმავე მნიშვნელობის ფარგლებშია და overfit არ შეიმჩნევა.


## Convolutional Neural Networks

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets

### Experiment 1: 1-Layer-CNN-1-Layer-FCNN

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/k1lyvisq

ტრადიციულად ConvNet-ებზე ექსპერიმენტს ვიწყებ უმარტივესი მოდელით, რომელშიც შედის 1 CNN-ის ფენა და 1 FCNN-ის ფენა.

channel-ების რაოდენობად პირველ ფენაში ავიღე 32.

```
Correct: 10283 / 20096, Train Accuracy: 0.5116938694267515
Validation: 1587 / 4306, Validation Accuracy: 0.3685555039479796
Epoch training time: 7.544518709182739 seconds
```

მოდელის 5 ეპოქიანი გაწვრთნის მერე მივიღე FCNN-ებთან სადარი შედეგი, მხოლოდ 4-6% ით ჩამორჩებოდა თავიდანვე, FCNN-ების საუკეთესო მოდელს, რაც მოსალოდნელიცაა CNN-ების კომპლექსურობიდან გამომდინარე.

### Experiment 2: 1-Layer-CNN-64-Channels-1-Layer-FCNN

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/74v4bo1n

ამ ექსპერიმენტში გადავწყვიტე 32 ჩენელის მაგივრად, პირველ ფენაში ამეღო 64 ჩენელი და დანარჩენი იგივე დამეტოვებინა წინა ექსპერიმენტის მოდელში.


```
Correct: 10333 / 20096, Train Accuracy: 0.5141819267515924
Validation: 1669 / 4306, Validation Accuracy: 0.387598699489085
Epoch training time: 9.305160760879517 seconds
-------------------------------------------

Total training time: 47.83097863197327 seconds
```

შედეგად 2% კი გაუმჯობესდა ვალიდაციის ქულა, თუმცა ეს ცვლილება ამ დროისთვის საკმაოდ მინიმალურია, ჩენელების 2-ჯერ გაზრდისათვის.

თეორიულად შეგვიძლია უსასრულოდ ვზარდოთ მსგავსი ჰიპერ-პარამეტრები და უკეთესი შედეგი უნდა მივიღოთ ყოველ ჯერზე, მაგრამ ყოველი გაზრდის შემდეგ ქულის გაუმჯობესება უფრო და უფრო მიზერული ხდება და ამიტომაც კარგი ბალანსის პოვნაა საჭირო.

CNN-ის პირველი ფენის ჩენელები დაახლოებით შეესაბამება თვალით შესამჩნევ განსხვავებულ feature-ებს, რომელიც სურათზეა და ეს ფენები ამ feature-ებს იმახსოვრებენ. 48x48 სურათზე, როგორც ჩანს 32 ჩენელის კარგად ართმევს დავალებას თავს.

### Experiment 3: 2-Layer-CNN-1-Layer-FCNN

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/oimlwd0e

მეორე ვარიანტით შესაძლებელია, რომ ისევ ბევრი ჩენელი ავიღოთ პირველ ფენაზე და მეორე ფენაზე უფრო ნაკლები. ეს იმის ანალოგიაა, რომ თავიდან ბევრი feature დავიჭიროთ სურათში და შემდეგ ამ feature-ების კომბინაციისაგან გავაკეთოთ გარკვეული დასკვნები.

ამ შემთხვევაში უკვე მივიღებთ 2 ფენიან CNN-ს, მაგრამ როდესაც ეს ექსპერიმენტი ვცადე, მოდელი 5 ეპოქაშივე overfitted გახდა.

```
Correct: 19122 / 20096, Train Accuracy: 0.9515326433121019
Validation: 1548 / 4306, Validation Accuracy: 0.35949837436135623
Epoch training time: 8.876072883605957 seconds
-------------------------------------------

Total training time: 42.23257660865784 seconds
```

ამ პრობლემის გადასაჭრელად გადავწყვიტე სხვადასხვა რეგულარიზაციის ტექნიკები გამომეყენებინა CNN-ებისათვის.


### Experiment 4: 2-Layer-CNN-1-Layer-FCNN-MaxPool

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/lg8ultz5


MaxPool რეგულარიზაციის მთავარი იდეაა, რომ მხოლოდ მკვეთრი feature-ები დაიჭიროს სურათში და მართლაც overfitted მოდელი აღარ მივიღე, როდესაც წინა მოდელს MaxPool ფენები დავუმატე train და validation ქულები 5 ეპოქაში ერთმანეთის თანაბარი დარჩა.

```
Correct: 4997 / 20096, Train Accuracy: 0.24865644904458598, Total Loss: 3664.909227013588, Avg Loss: 0.18237008494295323
Validation: 1067 / 4306, Validation Accuracy: 0.24779377612633535, Total Loss: 3664.909227013588
Epoch training time: 7.091748952865601 seconds
-------------------------------------------

Total training time: 33.16809439659119 seconds
```

### Experiment 6: 3-Layer-CNN-1-Layer-FCNN-MaxPool

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/0jzl26vd

კარგი რეგულაციის შედეგად საშუალება მეძლევა ხოლმე, მოდელის კომპლექსურობა გავზარდო და დავაკვირდე overfit-ს.

```
Correct: 5015 / 20096, Train Accuracy: 0.24955214968152867
Validation: 1102 / 4306, Validation Accuracy: 0.25592196934509986
Epoch training time: 8.747547388076782 seconds
-------------------------------------------

Total training time: 45.80154347419739 seconds
```

როგორც ხედავთ 3 ფენიან CNN-ზეც კი დამაკმაყოფილებელ შედეგებს იძლევა MaxPool, მაგრამ ახლა მიწევს გაცილებით უფრო მეტი ეპოქა ვწვრთნა მოდელი, რათა კარგი შედეგი მივიღო ვალიდაციაზე.

### Experiment 7: 3-Layer-CNN-2-Layer-FCNN-Batchnorm

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/6q90sww0

აღსანიშნავია, რომ წინა ექსპერიმენტში MaxPool რეგულარიზაცია მაქვს მხოლოდ CNN-ფენებზე და ეს იმას არ გამორიცხავს, რომ FCNN ფენები CNN-ის დაგენერირებულ შედეგზე და-overfit-დეს.

ამ მოსაზრების გადასამოწმებლად კიდევ 1 ფენა დავამატე FCNN-ს, რომელიც CNN ფენების შემდეგ არის განლაგებული. FCNN პასუხისმგებელია ამ შემთხვევაში CNN-ის მიერ დაგენერირებული შედეგების ინტერპრეტირებაზე.

```
Correct: 13075 / 20096, Train Accuracy: 0.6506269904458599
Validation: 2113 / 4306, Validation Accuracy: 0.490710636321412
Epoch training time: 71.53820562362671 seconds
-------------------------------------------

Total training time: 353.536416053772 seconds
```

ამ ცვლილებამ თითქმის 50% აიყვანა ვალიდაცია, რაც კარგი შედეგია, თუმცა ძალიან გაზრდა training ქულის არაა სასურველი. აქ უკვე ვფიქრობ, რომ შესაძლებელია FCNN ფენასაც დავამატო რეგულარიზაციის საშუალებები, მაგალითად Dropout.


### Experiment 8: 2-Layer-CNN-2-Layer-FCNN-Batchnorm-IncChannels

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/rxcgg9ht

აქამდე ექსპერიმენტებში CNN ქსელს ვაწყობდი ისე, რომ პირველ ფენებზე ბევრი ჩენელი იყო და შემდეგ ფენებზე უფრო და უფრო მცირდებოდა ჩენელების რაოდენობა.

ახლა გამიჩნდა იდეა, რომ გამეტესტა მეორე გზაც. თავდაპირველ ფენებზე დავსვათ შედარებით მცირე რაოდენობის ჩენელები, თუმცა მათი შედეგების ინტერპრეტაციას დავუთმოთ უფრო მეტი ჩენელი.

აქედან გამომდინარე, პირველ ფენაზე დავსვი 32 ჩენელი, ხოლო მეორეზე 64 ჩენელი და, როგორც წინა მოდელში, შედეგები გადავეცი 2 ფენიან FCNN-ს.

```
Correct: 13339 / 20096, Train Accuracy: 0.6637639331210191
Validation: 1962 / 4306, Validation Accuracy: 0.4556432884347422
Epoch training time: 120.70689129829407 seconds
-------------------------------------------

Total training time: 611.9506108760834 seconds
```

ყველაზე მკვეთრი ცვლილება იყო გაწვრთნის დროში, ასეთ მოდელებს გაცილებით უფრო დიდი ხანი სჭირდებათ training-ისთვის.

თუმცა ვალიდაციის ქულაშიც არ მოიტანა დიდი გაუმჯობესება მეორეული feature-ების გაშიფვრის მიდგომამ.

### Experiment 9: 2-Layer-CNN-3-Layer-FCNN-IncChannels-MaxPool

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/oicknnei

### Experiment 10: 2-Layer-CNN-3-Layer-FCNN-NoRegulation

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/9fkrz9on

შემდგომ ექსპერიმენტებში კიდევ მოვსინჯე მსგავსი ტექნიკა სხვადასხვა ცვლილებებით, მაგალითად MaxPool რეგულარიზაცია დავამატე და შემდეგ რეგულარიზაციის გარეშე ვცადე, მაგრამ მთავარი პრობლემა ის იყო, რომ

1. უკეთეს შედეგს არ მაძლევდა წინა მოდელებთან შედარებით
1. ძალიან დიდი ხანი უნდოდა მოდელის გაწვრთნას
1. Google Colab-ის სესია ხშირად იქრეშებოდა, რადგან GPU-ს VRAM არ ჰყოფნიდა weight და bias პარამეტრების შესანახად.

ამ ყველაფრიდან გამომდინარე, გადავწყვიტე ისევ ჯერ დიდი ზომის ჩენელები დამესვა და შემდეგ პატარა ზომის.


### Experiment 11: 2-Layer-CNN-2-Layer-FCNN-MaxPool-HighChannel

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/x27zq6sc

საბოლოო მოდელისაკენ სიარული ამ ექსპერიმენტში იწყება. როგორც ზემოთ ვთქვი ამ მოდელში ჯერ დიდი ზომის 256 ჩენელიანი layer დავსვი და მერე 128 ჩენელიანი layer და შემდეგ 2 ფენიან FCNN-ს გადავეცი ეს შედეგები.

```
Epoch 9
Iteration 0: Loss 0.011970708146691322
Iteration 100: Loss 0.018528463318943977
Correct: 20051 / 20096, Train Accuracy: 0.9977607484076433, Total Loss: 2.86141307791695, Avg Loss: 0.0001423871953581285
Validation: 1672 / 4306, Validation Accuracy: 0.3882954017649791, Total Loss: 2.86141307791695
Epoch training time: 69.11424970626831 seconds
-------------------------------------------

Total training time: 674.2054390907288 seconds
```

მოდელი გავწვრთენი 10 ეპოქის განმავლობაში და მივიღე უკიდურესი overfit, მიუხედავად იმისა, რომ ვიყენებდი MaxPool-ს.


### Experiment 12: 3-Layer-CNN-2-Layer-FCNN-Batchnorm-MaxPool

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/st0g981j

წინა მოდელის შემდეგი იტერაცია გამოირჩეოდა იმით, რომ დავუმატე BatchNorm CNN-ის ყველა ფენას და CNN ფენების რაოდენობა გავზარდე 3-მდე.

იდეა ის იყო, რომ დამატებითი რეგულარიზაცია Batchnorm-ისაგან მეგონა შეამცირებდა overfit-ს და ამავდროულად რადგან ახლა deep ქსელი იყო weight-ების ნორმალიზაციაც კარგი იქნებოდა.


```
Iteration 100: Loss 0.12169039994478226
Correct: 19976 / 20096, Train Accuracy: 0.9940286624203821, Total Loss: 10.718614540994167, Avg Loss: 0.0005333705484173053
Validation: 2235 / 4306, Validation Accuracy: 0.5190431955411055, Total Loss: 10.718614540994167
Epoch training time: 6.333305358886719 seconds
-------------------------------------------

Total training time: 177.1694495677948 seconds
```

მივიღე გაცილებით უფრო overfitted მოდელი, მაგრამ რადგან ვალიდაციის ქულა 51%-მდე ავიდა პირველად, ვიფიქრე, რომ რეგულაციების დამატების შედეგად შემეძლო მსგავსი შედეგების მიღება overfit-ის გარეშე.


### Experiment 13: 3-Layer-CNN-2-Layer-FCNN-Batchnorm-MaxPool-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/st0g981j

სწორედ ეს გავაკეთე შემდეგ იტერაციაზე, უბრალოდ დავუმატე dropout რეგულარიზაცია FCNN ფენებს `dropout=0.5` ჰიპერ-პარამეტრით, რადგან ეჭვი მქონდა, რომ სწორედ FCNN ნაწილი იწვევდა overfit-ს.


```
Epoch 9
Iteration 0: Loss 1.096341609954834
Iteration 100: Loss 1.1263242959976196
Correct: 11587 / 20096, Train Accuracy: 0.5765824044585988, Total Loss: 115.99165272712708, Avg Loss: 0.005771877623762294
Validation: 2151 / 4306, Validation Accuracy: 0.4995355318160706, Total Loss: 115.99165272712708
Epoch training time: 6.196078300476074 seconds
-------------------------------------------

Total training time: 59.87181091308594 seconds
```

ჩემი ვარაუდი სწორი გამოდგა და overfit შემცირდა საგრძნობლად თან ისე, რომ validation ქულა ძალიან არ გაფუჭდა.

### Experiment 14: 3-Layer-CNN-3-Layer-FCNN-Batchnorm-MaxPool-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/j2xbxier


შემდეგ ექსპერიმენტში შევეცადე გამეზარდა FCNN ფენების რაოდენობა და ამ შემთხვევაში აღმოვაჩინე, რომ overfit-ის შანსი იზრდება dropout-ი როცა მაქვს ჩართული მაშინაც კი, თუმცა ვალიდაციის ქულა 52%-დე გავზარდე.

### Experiment 15: 2-Layer-CNN-3-Layer-FCNN-Batchnorm-MaxPool-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/ziyeg6s3


შემდეგ იტერაციაზე 2-მდე შევამცირე CNN-ების ფენები.

```
Epoch 14
Iteration 0: Loss 1.0313057899475098
Iteration 100: Loss 0.9984095096588135
Correct: 11681 / 20096, Train Accuracy: 0.5812599522292994, Total Loss: 112.68670296669006, Avg Loss: 0.005607419534568574
Validation: 2177 / 4306, Validation Accuracy: 0.5055736182071529, Total Loss: 112.68670296669006
Epoch training time: 20.92659282684326 seconds
-------------------------------------------

Total training time: 328.5661315917969 seconds
```

ამ ცვლილებამ ვალიდაციის accuracy დააგდო ისევ 2% და საბოლოოდ გადავწყვიტე, რომ დამეტოვებინა 3 ფენა CNN-ისათვის.


### Experiment 16: 3-Layer-CNN-3-Layer-FCNN-Batchnorm-MaxPool-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/8diujv43

ბევრი ფიქრის შემდეგ, მე-14 ექსპერიმენტი ავიღე და ყველა CNN ჩენელის ფენების რაოდენობა გავაორმაგე ამ ექსპერიმენტში. შედეგად მივიღე უფრო მეტი ექსპრესიულობა და ვალიდაციის ქულა თითქმის 55% გახდა.

ეს ყველაფერი იმის დამსახურებაა, რომ ამ მოდელში გამოვიყენე მთლიანი ცოდნა, რაც ამ ექსპერიმენტების განმავლობაში მივიღე. ერთდროულად ვიყენებ BatchNorm, MaxPool-სა და Dropout-ს ნეირონული ქსელის სხვადასხვა ნაწილებზე.

მკაცრმა რეგულარიზაციამ საშუალება მომცა, რომ მოდელის კომპლექსურობა გამეზარდა overfit-ის გარეშე.


### Experiment 17: 4-Layer-CNN-3-Layer-FCNN-Batchnorm-MaxPool-Dropout-50

https://wandb.ai/vvaza22-free-university-of-tbilisi/Assignment4-ConvNets/runs/vvjfgun2

ვცადე მე-17 ექსპერიმენტიც, რომელშიც 4-მდე ავწიე CNN ფენების რაოდენობა, თუმცა ამან, პირიქით, შეამცირა validation ქულა 15 ეპოქაში და გადავწყვიტე, რომ მე-16 ექსპერიმენტის შედეგად მიღებული მოდელი ამერჩია საუკეთესოდ.


# Inference

გადმოვწერე wandb-დან უკან onnx ფაილი, რომელიც შეესაბამებოდა მე-16 ექსპერიმენტს ConvNets-ებიდან.

გავუშვი სატესტოდ შენახულ მონაცემებზე და მივიღე შედეგი:

```
Number of correct answers: 2314 / 4307
Model Accuracy: 0.537264917576039
```

იქიდან გამომდინარე, რომ ძალიან ახლოს არის predicted 55%-თან, რომელიც ვალიდაციის set-ზე მივიღე, შემიძლია ვთქვა, რომ ჩემმა შრომამ რეგულარიზაციაზე გაამართლა და მივიღე ისეთი მოდელი, რომელიც სანდომიანად 53% ფარგლებში სწორად განასხვავებს ადამიანის ემოციებს.

აღსანიშნავია, რომ Kaggle-ზე საუკეთესო შედეგი, რომელმაც competition-ში გაიმარჯვა, არის მხოლოდ 71%.
