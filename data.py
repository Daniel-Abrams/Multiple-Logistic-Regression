import pandas as pd
import numpy as np
file_path = 'data.csv'
df = pd.read_csv(file_path)


# Parameter Data
gender = df['gender'].tolist()
age = df['age'].tolist()
relationship = df['marital'].tolist()
parent = df['par'].tolist()
education = df['educ2'].tolist()
employment = df['emplnw'].tolist()
race = df['racem1'].tolist()
income = df['income'].tolist()
political = df['party'].tolist()
# socialMedia = df['socialmed'].tolist()




# Social Media Data
Twitter = df['web1A'].tolist()
Instagram = df['web1B'].tolist()
FaceBook = df['web1C'].tolist()
Snapchat = df['web1D'].tolist()
YouTube = df['web1E'].tolist()
WhatsApp = df['web1F'].tolist()
Pinterest = df['web1G'].tolist()
LinkedIn = df['web1H'].tolist()
Reddit = df['web1I'].tolist()
TikTok = df['web1J'].tolist()
Nextdoor = df['web1K'].tolist()


# Training Data
genderTraining = gender[0:1350]
ageTraining = age[0:1350]
relationshipTraining = relationship[0:1350]
parentTraining = parent[0:1350]
educationTraining = education[0:1350]
employmentTraining = employment[0:1350]
raceTraining = race[0:1350]
incomeTraining = income[0:1350]
politicalTraining = political[0:1350]
biases = [1] * 1350
TrainingData = [genderTraining,ageTraining,relationshipTraining,parentTraining,educationTraining,employmentTraining,raceTraining,incomeTraining,politicalTraining,biases]

TwitterTraining = Twitter[0:1350]
InstagramTraining = Instagram[0:1350]
FaceBookTraining = FaceBook[0:1350]
SnapchatTraining = Snapchat[0:1350]
YouTubeTraining = YouTube[0:1350]
WhatsAppTraining = WhatsApp[0:1350]
PinterestTraining = Pinterest[0:1350]
LinkedInTraining = LinkedIn[0:1350]
RedditTraining = Reddit[0:1350]
TikTokTraining = TikTok[0:1350]
NextdoorTraining = Nextdoor[0:1350]
socialMediaTrainingData = [TwitterTraining,InstagramTraining,FaceBookTraining,SnapchatTraining,YouTubeTraining,WhatsAppTraining,PinterestTraining,LinkedInTraining,RedditTraining,TikTokTraining,NextdoorTraining]

# Test Data
genderTest = gender[1350:1501]
ageTest = age[1350:1501]
relationshipTest = relationship[1350:1501]
parentTest = parent[1350:1501]
educationTest = education[1350:1501]
employmentTest = employment[1350:1501]
raceTest = race[1350:1501]
incomeTest = income[1350:1501]
politicalTest = political[1350:1501]


TwitterTest = Twitter[1350:1500]
InstagramTest = Instagram[1350:1500]
FaceBookTest = FaceBook[1350:1500]
SnapchatTest = Snapchat[1350:1500]
YouTubeTest = YouTube[1350:1500]
WhatsAppTest = WhatsApp[1350:1500]
PinterestTest = Pinterest[1350:1500]
LinkedInTest = LinkedIn[1350:1500]
RedditTest = Reddit[1350:1500]
TikTokTest = TikTok[1350:1500]
NextdoorTest = Nextdoor[1350:1500]
socialMediaTestData = [TwitterTest,InstagramTest,FaceBookTest,SnapchatTest,YouTubeTest,WhatsAppTest,PinterestTest,LinkedInTest,RedditTest,TikTokTest,NextdoorTest]

# Normalize the data
def normalizeData():
    for i in range(len(genderTraining)):
        genderTraining[i]  = (genderTraining[i]-1)/(3-1.0)
    
    for i in range(len(genderTest)):
        genderTest[i]  = (genderTest[i]-1)/(3-1.0)

    for i in range(len(ageTraining)):
        ageTraining[i]  = (ageTraining[i]-18)/(96-18.0)
    
    for i in range(len(ageTest)):
        ageTest[i]  = (ageTest[i]-1)/(96-18.0)

    for i in range(len(relationshipTraining)):
        relationshipTraining[i]  = (relationshipTraining[i]-1)/(6-1.0)
    
    for i in range(len(relationshipTest)):
        relationshipTest[i]  = (relationshipTest[i]-1)/(6-1.0)

    for i in range(len(parentTraining)):
        parentTraining[i]  = (parentTraining[i]-1)/(2-1.0)

    for i in range(len(parentTest)):
        parentTest[i]  = (parentTest[i]-1)/(2-1.0)

    for i in range(len(educationTraining)):
        educationTraining[i]  = (educationTraining[i]-1)/(8-1.0)
    
    for i in range(len(educationTest)):
        educationTest[i]  = (educationTest[i]-1)/(8-1.0)

    for i in range(len(employmentTraining)):
        employmentTraining[i]  = (employmentTraining[i]-1)/(8-1.0)
        
    for i in range(len(employmentTest)):
        employmentTest[i]  = (employmentTest[i]-1)/(8-1.0)

    for i in range(len(raceTraining)):
        raceTraining[i]  = (raceTraining[i]-1)/(7-1.0)
     
    for i in range(len(raceTest)):
        raceTest[i]  = (raceTest[i]-1)/(7-1.0)

    for i in range(len(incomeTraining)):
        incomeTraining[i]  = (incomeTraining[i]-1)/(9-1.0)
    
    for i in range(len(incomeTest)):
        incomeTest[i]  = (incomeTest[i]-1)/(9-1.0)

    for i in range(len(politicalTraining)):
        politicalTraining[i]  = (politicalTraining[i]-1)/(2-1.0)
    
    for i in range(len(politicalTest)):
        politicalTest[i]  = (politicalTest[i]-1)/(2-1.0)


# 2D Matrix to hold the weights. The rows are the Paramters. Columns are the clasese.
weights = np.random.uniform(-0.1, 0.1, size=(10, 11))
