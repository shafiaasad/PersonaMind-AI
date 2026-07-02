import os
import re
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler



warnings.filterwarnings("ignore", category=ConvergenceWarning)

APP_TITLE = "PersonaMind AI 2.0"
DATA_DIR = "data"
DATA_PATH = os.path.join(DATA_DIR, "sample_persona_dataset.csv")
ASSET_DIR = "assets"
HERO_IMAGE = os.path.join(ASSET_DIR, "persona-hero.svg")
PROFILE_IMAGE = os.path.join(ASSET_DIR, "profile-lab.svg")
LOGO_IMAGE = os.path.join(ASSET_DIR, "logo.png")
RANDOM_STATE = 42


SUBJECT_OPTIONS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Deep Learning",
    "Data Science",
    "Programming",
    "Mathematics",
    "Statistics",
    "Cybersecurity",
    "Computer Networks",
    "Database Systems",
    "Software Engineering",
    "Web Development",
    "Mobile App Development",
    "Robotics",
    "Natural Language Processing",
    "Computer Vision",
    "Cloud Computing",
    "Business Studies",
    "Psychology",
    "English Communication",
    "Research Methods",
]

INTEREST_OPTIONS = [
    "Robotics",
    "Chatbots",
    "Data analysis",
    "AI tools",
    "Game development",
    "Mobile apps",
    "Web apps",
    "Cybersecurity",
    "Ethical hacking",
    "Teaching",
    "Public speaking",
    "Business startups",
    "Graphic design",
    "UI/UX design",
    "Research",
    "Writing",
    "Problem solving",
    "Team leadership",
    "Freelancing",
    "Automation",
    "Cloud technologies",
    "Digital marketing",
]

SKILL_OPTIONS = [
    "python",
    "java",
    "c++",
    "javascript",
    "html",
    "css",
    "sql",
    "machine learning",
    "deep learning",
    "nlp",
    "data analysis",
    "statistics",
    "visualization",
    "excel",
    "power bi",
    "communication",
    "presentation",
    "teamwork",
    "leadership",
    "research",
    "writing",
    "critical thinking",
    "problem solving",
    "debugging",
    "databases",
    "linux",
    "networking",
    "security",
    "marketing",
    "planning",
    "creativity",
]

CAREER_DATABASE = {
    "AI Engineer": {
        "skills": ["python", "machine learning", "deep learning", "nlp", "math", "problem solving"],
        "environment": "Research Environment",
        "description": "Builds intelligent systems, models, assistants, and predictive AI tools.",
    },
    "Data Scientist": {
        "skills": ["python", "statistics", "data analysis", "machine learning", "visualization", "sql"],
        "environment": "Corporate Environment",
        "description": "Finds patterns in data and turns them into useful decisions.",
    },
    "Software Engineer": {
        "skills": ["programming", "python", "java", "problem solving", "databases", "debugging"],
        "environment": "Remote Work",
        "description": "Designs and develops reliable software products and applications.",
    },
    "Teacher": {
        "skills": ["communication", "teaching", "patience", "planning", "presentation", "empathy"],
        "environment": "Team Work",
        "description": "Explains concepts, guides learners, and builds confidence.",
    },
    "Business Analyst": {
        "skills": ["communication", "analysis", "excel", "problem solving", "documentation", "strategy"],
        "environment": "Corporate Environment",
        "description": "Connects business needs with practical technology and process solutions.",
    },
    "Entrepreneur": {
        "skills": ["leadership", "risk taking", "creativity", "communication", "marketing", "planning"],
        "environment": "Startup Environment",
        "description": "Creates new products, teams, and business opportunities.",
    },
    "Researcher": {
        "skills": ["research", "writing", "statistics", "critical thinking", "experiments", "reading"],
        "environment": "Research Environment",
        "description": "Investigates ideas deeply and produces evidence-based knowledge.",
    },
    "Cybersecurity Analyst": {
        "skills": ["networking", "security", "linux", "problem solving", "risk analysis", "python"],
        "environment": "Corporate Environment",
        "description": "Protects systems, detects threats, and improves digital safety.",
    },
}

CAREER_OPTIONS = list(CAREER_DATABASE.keys()) + [
    "Machine Learning Engineer",
    "NLP Engineer",
    "Computer Vision Engineer",
    "Cloud Engineer",
    "UI/UX Designer",
    "Product Manager",
    "Digital Marketer",
    "Freelancer",
]

CLUSTER_NAMES = {
    0: "Analytical Thinkers",
    1: "Creative Leaders",
    2: "Balanced Learners",
    3: "Practical Problem Solvers",
}


@dataclass
class UserProfile:
    name: str
    age: int
    gender: str
    favorite_subjects: str
    interests: str
    current_skills: str
    career_goal: str
    study_hours: float
    stress_level: int
    communication: int
    leadership: int
    creativity: int
    problem_solving: int
    risk_taking: int
    teamwork: int
    work_environment: str
    self_text: str
    q_social_energy: str
    q_learning_preference: str
    q_decision_style: str
    q_pressure_response: str
    q_project_role: str


def create_sample_dataset(path=DATA_PATH, rows=960):
    """Create a high-accuracy synthetic dataset with clear career patterns."""
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_STATE)

    career_profiles = {
        "AI Engineer": [22, 6.8, 4, 5, 7, 5, 9, 5, 6],
        "Data Scientist": [23, 6.2, 4, 6, 5, 4, 8, 3, 5],
        "Software Engineer": [22, 5.5, 5, 4, 6, 4, 8, 4, 5],
        "Teacher": [24, 4.8, 3, 9, 6, 6, 5, 2, 9],
        "Business Analyst": [24, 4.5, 5, 8, 5, 7, 7, 4, 8],
        "Entrepreneur": [25, 4.2, 6, 8, 9, 9, 6, 9, 7],
        "Researcher": [24, 7.2, 3, 4, 6, 3, 9, 2, 4],
        "Cybersecurity Analyst": [23, 5.8, 6, 4, 4, 4, 9, 6, 5],
    }
    career_text = {
        "AI Engineer": "python machine learning deep learning nlp artificial intelligence model neural network automation",
        "Data Scientist": "data analysis statistics visualization sql pandas numpy machine learning dashboard insights",
        "Software Engineer": "programming software debugging databases web app python java problem solving code",
        "Teacher": "teaching communication presentation classroom planning teamwork explanation students learning",
        "Business Analyst": "business analysis excel communication documentation strategy requirements corporate decision making",
        "Entrepreneur": "startup leadership creativity risk taking marketing business planning innovation product",
        "Researcher": "research writing experiments statistics reading critical thinking academic paper study",
        "Cybersecurity Analyst": "security networking linux risk analysis ethical hacking cyber threat protection python",
    }
    learning_map = {
        "AI Engineer": "Practical Learner",
        "Data Scientist": "Reading/Writing Learner",
        "Software Engineer": "Practical Learner",
        "Teacher": "Auditory Learner",
        "Business Analyst": "Visual Learner",
        "Entrepreneur": "Practical Learner",
        "Researcher": "Reading/Writing Learner",
        "Cybersecurity Analyst": "Practical Learner",
    }
    cols = ["age", "study_hours", "stress_level", "communication", "creativity", "leadership", "problem_solving", "risk_taking", "teamwork"]
    records = []
    per_class = max(80, rows // len(career_profiles))

    for career, values in career_profiles.items():
        for _ in range(per_class):
            record = {}
            for col, val in zip(cols, values):
                if col == "age":
                    record[col] = int(np.clip(round(rng.normal(val, 1.4)), 17, 31))
                elif col == "study_hours":
                    record[col] = round(float(np.clip(rng.normal(val, 0.45), 1, 8)), 1)
                else:
                    record[col] = int(np.clip(round(rng.normal(val, 0.75)), 1, 10))

            if record["communication"] >= 7 and record["teamwork"] >= 7:
                personality = "Extrovert"
            elif record["communication"] <= 4 and record["teamwork"] <= 5:
                personality = "Introvert"
            else:
                personality = "Ambivert"

            success_score = (
                record["study_hours"] * 8
                + record["problem_solving"] * 7
                + record["communication"] * 4
                + record["leadership"] * 4
                + record["creativity"] * 4
                + record["teamwork"] * 3
                - record["stress_level"] * 3
            )
            success = "High" if success_score >= 125 else "Medium" if success_score >= 80 else "Low"
            record.update(
                {
                    "personality_type": personality,
                    "career_category": career,
                    "learning_style": learning_map[career],
                    "success_level": success,
                    "text": f"I am interested in {career_text[career]}. My goal is {career.lower()} and I enjoy practical projects related to {career.lower()}.",
                }
            )
            records.append(record)

    data = pd.DataFrame(records)
    data.to_csv(path, index=False)
    try:
        data.to_excel(os.path.join(DATA_DIR, "sample_persona_dataset.xlsx"), index=False)
    except Exception:
        pass

def load_dataset():
    if not os.path.exists(DATA_PATH):
        create_sample_dataset()
    required_columns = {
        "age",
        "study_hours",
        "stress_level",
        "communication",
        "creativity",
        "leadership",
        "problem_solving",
        "risk_taking",
        "teamwork",
        "personality_type",
        "career_category",
        "learning_style",
        "success_level",
        "text",
    }
    try:
        df = pd.read_csv(DATA_PATH)
        if not required_columns.issubset(df.columns) or len(df) < 50:
            create_sample_dataset()
            df = pd.read_csv(DATA_PATH)
        numeric_cols = [
            "age",
            "study_hours",
            "stress_level",
            "communication",
            "creativity",
            "leadership",
            "problem_solving",
            "risk_taking",
            "teamwork",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=numeric_cols + ["career_category", "text"])
        if len(df) < 50:
            create_sample_dataset()
            df = pd.read_csv(DATA_PATH)
        return df
    except Exception:
        create_sample_dataset()
        return pd.read_csv(DATA_PATH)



def train_models(df):
    feature_cols = [
        "age",
        "study_hours",
        "stress_level",
        "communication",
        "creativity",
        "leadership",
        "problem_solving",
        "risk_taking",
        "teamwork",
    ]
    X = df[feature_cols]
    y = df["career_category"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=RANDOM_STATE, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train_scaled, y_train)
    knn_pred = knn.predict(X_test_scaled)

    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", max_iter=1200, random_state=RANDOM_STATE, early_stopping=True)
    mlp.fit(X_train_scaled, y_train)
    mlp_pred = mlp.predict(X_test_scaled)

    text_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=500, stop_words="english")),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]
    )
    text_pipeline.fit(df["text"].fillna(""), df["career_category"])

    kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=10)
    kmeans.fit(scaler.transform(X))

    return {
        "feature_cols": feature_cols,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "knn": knn,
        "mlp": mlp,
        "text_pipeline": text_pipeline,
        "kmeans": kmeans,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "knn_pred": knn_pred,
        "mlp_pred": mlp_pred,
        "knn_accuracy": accuracy_score(y_test, knn_pred),
        "mlp_accuracy": accuracy_score(y_test, mlp_pred),
    }



def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()



def skill_tokens(text):
    return {token.strip().lower() for token in re.split(r"[,;\n]+", text) if token.strip()}



def profile_to_features(profile):
    return pd.DataFrame(
        [
            {
                "age": profile.age,
                "study_hours": profile.study_hours,
                "stress_level": profile.stress_level,
                "communication": profile.communication,
                "creativity": profile.creativity,
                "leadership": profile.leadership,
                "problem_solving": profile.problem_solving,
                "risk_taking": profile.risk_taking,
                "teamwork": profile.teamwork,
            }
        ]
    )



def closest_career_search(profile, top_n=5):
    current = skill_tokens(profile.current_skills)
    interests = clean_text(profile.interests + " " + profile.favorite_subjects + " " + profile.career_goal)
    results = []

    for career, info in CAREER_DATABASE.items():
        required = set(info["skills"])
        matched = current.intersection(required)
        missing = sorted(required - current)
        interest_bonus = sum(1 for word in career.lower().split() if word in interests)
        score = len(matched) * 15 + interest_bonus * 10
        score += min(profile.problem_solving, 10) if career in ["AI Engineer", "Data Scientist", "Software Engineer"] else 0
        score += min(profile.communication, 10) if career in ["Teacher", "Business Analyst", "Entrepreneur"] else 0
        score += min(profile.risk_taking, 10) if career == "Entrepreneur" else 0
        score = int(min(100, score))
        results.append(
            {
                "Career": career,
                "Compatibility": score,
                "Matched Skills": ", ".join(sorted(matched)) if matched else "None yet",
                "Missing Skills": ", ".join(missing) if missing else "No major gaps",
                "Environment": info["environment"],
            }
        )

    return pd.DataFrame(results).sort_values("Compatibility", ascending=False).head(top_n)



def rule_personality(profile):
    if profile.q_social_energy.startswith("I enjoy groups") or (
        profile.communication >= 8 and profile.teamwork >= 7
    ):
        return "Extrovert"
    if profile.q_social_energy.startswith("I prefer quiet") or (
        profile.communication <= 4 and profile.teamwork <= 5
    ):
        return "Introvert"
    return "Ambivert"



def rule_learning_style(profile):
    if "visual" in profile.q_learning_preference.lower():
        return "Visual Learner"
    if "reading" in profile.q_learning_preference.lower():
        return "Reading/Writing Learner"
    if "practice" in profile.q_learning_preference.lower():
        return "Practical Learner"
    return "Auditory Learner"



def category(value, low=40, high=70):
    if value < low:
        return "Low"
    if value < high:
        return "Medium"
    return "High"



def generate_predictions(profile, models):
    features = profile_to_features(profile)
    scaled = models["scaler"].transform(features)

    knn_career = models["label_encoder"].inverse_transform(models["knn"].predict(scaled))[0]
    nn_career = models["label_encoder"].inverse_transform(models["mlp"].predict(scaled))[0]

    text = clean_text(
        " ".join(
            [
                profile.self_text,
                profile.interests,
                profile.favorite_subjects,
                profile.career_goal,
                profile.current_skills,
            ]
        )
    )
    text_career = models["text_pipeline"].predict([text or "student learning technology"])[0]
    search_df = closest_career_search(profile, top_n=8)
    search_career = search_df.iloc[0]["Career"]

    votes = pd.Series([knn_career, nn_career, text_career, search_career])
    career = votes.mode().iloc[0]

    cluster_raw = int(models["kmeans"].predict(scaled)[0])
    cluster_name = CLUSTER_NAMES.get(cluster_raw, "Balanced Learners")

    leadership_score = int(np.clip(profile.leadership * 7 + profile.communication * 2 + profile.teamwork * 1, 0, 100))
    entrepreneurship_score = int(
        np.clip(profile.risk_taking * 4 + profile.creativity * 3 + profile.leadership * 2 + profile.communication, 0, 100)
    )
    productivity_score = int(np.clip(profile.study_hours * 11 + profile.problem_solving * 4 - profile.stress_level * 3, 0, 100))
    communication_score = int(np.clip(profile.communication * 10, 0, 100))
    innovation_score = int(np.clip(profile.creativity * 6 + profile.problem_solving * 3 + profile.risk_taking, 0, 100))
    success_probability = int(
        np.clip(
            productivity_score * 0.35
            + innovation_score * 0.2
            + communication_score * 0.15
            + leadership_score * 0.15
            + (100 - profile.stress_level * 8) * 0.15,
            0,
            100,
        )
    )
    burnout_points = int(np.clip(profile.stress_level * 8 - profile.study_hours * 3 + (10 - profile.teamwork) * 2, 0, 100))

    team_role = "Developer"
    if leadership_score >= 75:
        team_role = "Leader"
    elif profile.problem_solving >= 8:
        team_role = "Analyst"
    elif profile.creativity >= 8:
        team_role = "Designer"
    elif "research" in text or profile.study_hours >= 6:
        team_role = "Researcher"
    elif profile.teamwork >= 8:
        team_role = "Coordinator"

    hidden_scores = {
        "Research": profile.study_hours * 9 + profile.problem_solving * 4,
        "Leadership": leadership_score,
        "Creativity": profile.creativity * 10,
        "Technical Thinking": profile.problem_solving * 8 + profile.study_hours * 3,
        "Communication": communication_score,
        "Problem Solving": profile.problem_solving * 10,
    }
    hidden_talent = max(hidden_scores, key=hidden_scores.get)

    decision_style = "Balanced"
    if profile.q_decision_style.startswith("Facts"):
        decision_style = "Logical"
    elif profile.q_decision_style.startswith("Feelings"):
        decision_style = "Emotional"

    best_environment = CAREER_DATABASE.get(career, {}).get("environment", profile.work_environment)
    if profile.work_environment != "No preference":
        best_environment = profile.work_environment

    problem_solving_level = category(profile.problem_solving * 10)
    burnout_risk = category(burnout_points, low=35, high=65)
    risk_level = category(profile.risk_taking * 10)
    personality = rule_personality(profile)
    learning_style = rule_learning_style(profile)

    teammate = {
        "Leader": "Analyst or Developer",
        "Developer": "Coordinator or Designer",
        "Researcher": "Leader or Communicator",
        "Analyst": "Creative Designer",
        "Designer": "Technical Developer",
        "Coordinator": "Problem Solver",
    }.get(team_role, "Balanced teammate")

    ai_twin = (
        f"{profile.name or 'This user'} is a {personality.lower()}, {learning_style.lower()} with "
        f"{problem_solving_level.lower()} problem-solving ability and strong signs of {hidden_talent.lower()}. "
        f"The profile currently fits {career} best, with a {success_probability}% future success probability "
        f"if the recommended skill gaps are improved."
    )

    return {
        "personality": personality,
        "career": career,
        "knn_career": knn_career,
        "nn_career": nn_career,
        "text_career": text_career,
        "learning_style": learning_style,
        "search_df": search_df,
        "leadership_score": leadership_score,
        "entrepreneurship_score": entrepreneurship_score,
        "team_role": team_role,
        "burnout_risk": burnout_risk,
        "productivity_score": productivity_score,
        "communication_score": communication_score,
        "innovation_score": innovation_score,
        "problem_solving_level": problem_solving_level,
        "success_probability": success_probability,
        "best_environment": best_environment,
        "teammate": teammate,
        "hidden_talent": hidden_talent,
        "decision_style": decision_style,
        "risk_level": risk_level,
        "cluster_name": cluster_name,
        "ai_twin": ai_twin,
    }



def advisor(profile, predictions):
    career = predictions["career"]
    gaps = predictions["search_df"]
    career_row = gaps[gaps["Career"] == career]
    missing = career_row.iloc[0]["Missing Skills"] if not career_row.empty else gaps.iloc[0]["Missing Skills"]
    advice = [
        f"Career advice: focus on {career} because your profile shows a strong match with its required behavior pattern.",
        f"Study advice: keep a weekly routine of {max(2, int(profile.study_hours))} focused study hours and add one practical project.",
        f"Missing skills to improve: {missing}.",
    ]
    if predictions["burnout_risk"] == "High":
        advice.append("Wellbeing advice: reduce overload, take planned breaks, and avoid last-minute study pressure.")
    if predictions["communication_score"] < 60:
        advice.append("Communication advice: practice short presentations and group discussion once per week.")
    if predictions["leadership_score"] >= 75:
        advice.append("Leadership advice: take responsibility for planning, deadlines, and team coordination in projects.")
    return advice



def combine_choices(selected, custom_text=""):
    values = [str(item).strip() for item in selected if str(item).strip()]
    extras = [item.strip() for item in re.split(r"[,;\n]+", custom_text or "") if item.strip()]
    combined = []
    for item in values + extras:
        if item.lower() not in [existing.lower() for existing in combined]:
            combined.append(item)
    return ", ".join(combined)



def validate_profile(profile):
    errors = []
    if not profile.name.strip():
        errors.append("Name is required.")
    if profile.age < 10 or profile.age > 80:
        errors.append("Age must be between 10 and 80.")
    if len(skill_tokens(profile.favorite_subjects)) < 2:
        errors.append("Choose or write at least two favorite subjects.")
    if len(skill_tokens(profile.interests)) < 2:
        errors.append("Choose or write at least two interests.")
    if len(skill_tokens(profile.current_skills)) < 2:
        errors.append("Choose or write at least two current skills.")
    if not profile.career_goal.strip():
        errors.append("Career goal is required.")
    if len(profile.self_text.strip()) < 30:
        errors.append("Write at least 30 characters in the short paragraph about yourself.")
    if profile.study_hours == 0:
        errors.append("Study hours cannot be 0 for a meaningful prediction.")
    return errors

