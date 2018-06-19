# VR final project
## Code
- test.py : camera + facial feature detection(可以在code裡面調latency)
- 有一個dlib官網的model太大，我放在雲端，要執行test.py時再把她載下來。
- [model 連結](https://www.dropbox.com/s/mkwdt53c6krn8vw/shape_predictor_68_face_landmarks.dat?dl=1)：要用時把她載下來

- image.py : 偵測相片臉部位置，把臉部位置改成川劇面具，目前還沒有去掉川劇面具圖片的白色背景，可能程式判斷白色地方用原本pixel試試
    >使用方法
    > python3 image.py [image1 path] [image2 path]
    > EX : python3 image.py test.jpg show.jpg
- mask/
放面具圖的地方，就是從這裡random選的
- faceMorph.py :
就是之前說可以試試看讓面具效果變好的做法(image morphing)，可以試試看，但不知道效果如何

## 工作
- 蒐集更多京劇面具照片去掉白色部分放到城市跑跑看
- 目前random的出現面具，只要一沒感測到到下次感測到會換面具，所以可以用手擋住臉，接著就會出現不同面具
- 目前算是有一個階段成品，看要不要做做看face morphing