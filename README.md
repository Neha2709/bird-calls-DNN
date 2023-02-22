# AI Birdie 
A system for automated multimodal identification birds from visual and acoustic data, by employing machine learning processes. We believe that such a system would greatly improve the efficiency and effectiveness of the
birding community and bird monitoring process.

## Acoustic Module  ##

A bird activity detector based on logarithm of frame energy which will identify the part where the bird call is present and then that part will be extracted and then it will be converted to MFCC.

![MFCC](https://user-images.githubusercontent.com/46594515/220765147-2df96dce-1309-4822-ba1f-093b81424b42.png)
<br /><br /><br />
Layer wise representation for the acoustic model

![image](https://user-images.githubusercontent.com/46594515/220765568-de85ee77-7af9-4f48-b1f0-4ccec46bfca8.png)


## Generative adversarial network  ##

We have used WaveGAN as a novel approach to synthesise bird calls for species which were not showing great results. We have used this approach to strengthen our claim to augment datasets to improve accuracy of the overall model.

| Bird Species | Original | Generated | 
| :-------------- | :---------: | :----------: | 
| Rock Bunting | ![image](https://user-images.githubusercontent.com/46594515/220767412-8e101000-86a6-4a32-9019-88f138b6f99a.png) | ![image](https://user-images.githubusercontent.com/46594515/220767466-73565de6-34b2-4a03-8ece-b689e0b32966.png)|
| Great barbet | ![image](https://user-images.githubusercontent.com/46594515/220767594-7f3beaf8-dcd7-4919-9418-965c8373cd84.png) | ![image](https://user-images.githubusercontent.com/46594515/220767772-aa98fc0a-927c-41e2-979f-e96c2828ad68.png)|
