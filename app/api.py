from fastapi import FastAPI , File , UploadFile 
from fastapi.middleware.cors import CORSMiddleware 
import uvicorn 
import numpy as np 
from io import BytesIO 
from PIL import Image 
import torch
from CNN import CNN , idx_to_classes
from torchvision import transforms  

app = FastAPI() 

origins = [
    "http://localhost",
    "http://localhost:3000",
] 

app.add_middleware(
    CORSMiddleware, 
    allow_origins = origins,
    allow_credentials = True, 
    allow_methods = ["*"], 
    allow_headers = ["*"],
) 

CLASS_NAMES =  [ 
   "Apple : Apple scab" , "Apple : Apple Black Rot" , "Apple : Apple Cedar Rust" , "Apple : Apple Healthy" ,  
   "Blueberry healthy",
   "Cherry : Cherry Healthy" , "Cherry : Cherry Powdery mildew" , 
   "Corn : Cercospora leaf spot Gray leaf spot" , "Corn : Corn Common rust" , "Corn : Corn Healthy" ,  "Corn : Corn Northern Leaf Blight" ,
   "Grape : Black rot" , "Grape : Healthy " , "Grape: Leaf blight" , "Grape: Esca", 
   "Orange: Citrus Greening",
   "Peach: Bacterial Spot" , "Peach: Healthy" , 
   "Pepper : Bacterail Spot" , "Pepper : Healthy" , 
   "Potato : Late Blight" , "Potato : Early Blight" , "Potato: Healthy" ,
   "Raspberry : Healthy" , 
   "Soybean : Healthy" ,
   "Strawberry : Straberry healthy" , " Strawberry : Strawberry Leaf scorch" ,
   "Tomato: Bacterial Spot" , "Tomato: Early Blight" , "Tomato: Healthy" , "Tomato: Late Blight" , "Tomato: Leaf Mold" , "Tomato: Septoria Leaf Spot" , "Tomato: Spider Mite two spotted" , "Tomato: Target Spot" , "Tomato: Tomato mosaic virus" , "Tomato: TomatoYellow Leaf Curl Virus" 
]

def prediction(image_obj , idx_to_classes): 
    INPUT_DIM = 224 

    preprocess = transforms.Compose([ 
        transforms.Resize(INPUT_DIM) , 
        transforms.CenterCrop(INPUT_DIM) , 
        transforms.ToTensor() , 
        transforms.Normalize(mean=[0.485, 0.456, 0.406] , std=[0.229, 0.224, 0.225]) ,
    ]) 

    pretrained_model = CNN(38) 
    pretrained_model.load_state_dict( 
        torch.load('plant_disease_model.pt' , map_location=torch.device('cuda'))  
    ) 

    im = image_obj 
    im_preprocessed = preprocess(im) 
    batch_img_tensor = torch.unsqueeze(im_preprocessed , 0) 
    output = pretrained_model(batch_img_tensor)
    output = output.detach().numpy()  
    index = np.argmax(output) 
    predicted_class = idx_to_classes[index] 
    confidence = np.max(output[0])  
    return predicted_class , confidence*100


@app.get("/plants")
async def get_plants():
    return {"plants": CLASS_NAMES}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
): 
    image = Image.open(file.file)
    print("type" , type(image))
    result = prediction(image , idx_to_classes)
    return {
        "img_batch" : str(type(image)) ,
        "prediction": result[0], 
        "confidence": result[1]
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)