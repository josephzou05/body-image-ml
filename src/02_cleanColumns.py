import pandas as pd

# Load raw data
df = pd.read_csv("data/body_image_gym_survey.csv")

# Rename columns to camelCase
df = df.rename(columns={
    "1. Age": "age",
    "2. Gender": "gender",
    "3. How often do you go to the gym?": "gymFrequency",
    "4. What are your main fitness goals?": "fitnessGoals",
    "5. Do you track your workouts or progress (e.g., with a notebook, app, or photos)?": "trackProgress",
    "6. Do you follow gym/fitness influencers or creators?": "followInfluencers",
    "7. Do you compare your physique to others at the gym or online?": "comparePhysique",
    "8. I am satisfied with my body right now.": "bodySatisfactionStatement",
    "9. On a scale from 0 to 10, how satisfied are you with your physical appearance right now?": "bodySatisfactionScore",
    "10. I believe I need to gain more muscle to look good": "needMoreMuscle",
    "11. How often do you feel like your body is too small or not muscular enough, even when others say you look fit?": "feelTooSmall",
    "12. I feel pressure to look a certain way because of gym culture or fitness influencers": "gymPressure",
    "13. How often do you check your body in the mirror or take progress pictures to assess how you look?": "bodyChecking",
    "14. I avoid posting pictures because I don't think I look \"lean\" or \"big\" enough.": "avoidPostingPics"
})

# Save cleaned data
df.to_csv("outputs/cleanData.csv", index=False)

print("Cleaned data saved to outputs/cleanData.csv")
print("Columns:")
print(df.columns.tolist())


#target column: bodySatisfactionScore
#input columns: age, gender, gymFrequency, fitnessGoals, trackProgress, followInfluencers, 
# comparePhysique, bodySatisfactionStatement, needMoreMuscle, feelTooSmall, gymPressure, 
# bodyChecking, avoidPostingPics