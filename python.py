
#impotaing libraries
import pandas as pd 

#creating the dataa
data = {
    "date": [
        "2025-12-08",
        "2025-12-08",
        "2025-12-09",
        "2025-12-10",
        "2025-12-11",
        "2025-12-11"
    ],
    "topic": [
        "Course Introduction & Resources",
        "Python Basics",
        "Python Basics",
        "Python Practice",
        "Pandas",
        "Environment Setup (Miniconda)"
 ],
"duration_minutes": [
        120,
        120,
        240,
        240,
        240,
        120
    ],
    "difficulty": [
        2,
        2,
        2,
        2,
        4,
        4
    ],
    "notes": [
        "Videos and blogs overview",
        "Operators and syntax",
        "If else and data types",
        "Data structures and practice",
        "Data handling was difficult",
        "Installation and setup issues"
    ]
}
#creating dataframe
# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv("learning_log.csv", index=False)
print("learning_log.csv file created successfully")