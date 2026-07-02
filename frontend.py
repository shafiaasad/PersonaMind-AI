import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from backend import (
    APP_TITLE, ASSET_DIR, CAREER_DATABASE, CAREER_OPTIONS, HERO_IMAGE,
    INTEREST_OPTIONS, LOGO_IMAGE, PROFILE_IMAGE, SKILL_OPTIONS, SUBJECT_OPTIONS,
    UserProfile, advisor, combine_choices, load_dataset, train_models,
    skill_tokens, validate_profile, generate_predictions,
)

def page_config():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.markdown(
        """
        <style>
        :root {
            --pm-ink: #172033;
            --pm-muted: #667085;
            --pm-blue: #2563eb;
            --pm-teal: #0f766e;
            --pm-pink: #c026d3;
            --pm-gold: #b7791f;
            --pm-panel: #ffffff;
            --pm-line: #e7ebf2;
        }
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(37, 99, 235, .14), transparent 30%),
                radial-gradient(circle at 92% 20%, rgba(15, 118, 110, .12), transparent 28%),
                linear-gradient(135deg, #f8fbff 0%, #f6f7fb 50%, #fdf8ff 100%);
        }
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
        h1, h2, h3 {color: var(--pm-ink); letter-spacing: 0;}
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        }
        section[data-testid="stSidebar"] * {color: #f8fafc;}
        section[data-testid="stSidebar"] .stRadio label {
            background: rgba(255, 255, 255, .06);
            border-radius: 8px;
            padding: .25rem .45rem;
            margin-bottom: .18rem;
        }
        .pm-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 52%, #0f766e 100%);
            border: 1px solid rgba(255, 255, 255, .18);
            border-radius: 8px;
            padding: 1.35rem;
            color: #ffffff;
            box-shadow: 0 18px 50px rgba(15, 23, 42, .18);
        }
        .pm-hero h1, .pm-hero h2, .pm-hero h3, .pm-hero p {color: #ffffff;}
        .pm-hero .eyebrow {
            color: #bfdbfe;
            font-size: .85rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        .pm-card {
            background: rgba(255, 255, 255, .92);
            border: 1px solid var(--pm-line);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            box-shadow: 0 12px 26px rgba(16, 24, 40, 0.07);
            min-height: 112px;
        }
        .pm-card h4 {
            margin: 0 0 .35rem 0;
            color: #344054;
            font-size: 0.95rem;
        }
        .pm-card .big {
            font-size: 1.65rem;
            font-weight: 700;
            color: #111827;
            line-height: 1.15;
        }
        .pm-note {
            background: #eef6ff;
            border-left: 4px solid var(--pm-blue);
            padding: .85rem 1rem;
            border-radius: 6px;
        }
        .pm-section {
            background: rgba(255, 255, 255, .82);
            border: 1px solid var(--pm-line);
            border-radius: 8px;
            padding: 1rem;
            margin: .45rem 0 1rem 0;
        }
        .pm-pill {
            display: inline-block;
            background: #e0f2fe;
            color: #075985;
            border: 1px solid #bae6fd;
            border-radius: 999px;
            padding: .18rem .55rem;
            margin: .12rem;
            font-size: .82rem;
            font-weight: 650;
        }
        .pm-warning {
            border: 1px solid #fed7aa;
            background: #fff7ed;
            color: #9a3412;
            border-radius: 8px;
            padding: .8rem .95rem;
        }
        div[data-testid="stForm"] {
            background: rgba(255, 255, 255, .88);
            border: 1px solid var(--pm-line);
            border-radius: 8px;
            padding: 1rem;
            box-shadow: 0 12px 28px rgba(16, 24, 40, .06);
        }
        .stButton > button, .stFormSubmitButton > button {
            background: linear-gradient(135deg, #2563eb, #0f766e);
            color: white;
            border: 0;
            border-radius: 8px;
            font-weight: 700;
        }
        .stButton > button:hover, .stFormSubmitButton > button:hover {
            border: 0;
            color: white;
            filter: brightness(1.04);
        }
        div[data-testid="stMetricValue"] {font-size: 1.65rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )



def get_profile_from_state():
    return st.session_state.get("profile")


def get_predictions():
    return st.session_state.get("predictions")



def option_chips(items, limit=12):
    visible = list(items)[:limit]
    if not visible:
        return
    chips = "".join([f"<span class='pm-pill'>{item}</span>" for item in visible])
    st.markdown(chips, unsafe_allow_html=True)



def card(title, value, note=""):
    st.markdown(
        f"""
        <div class="pm-card">
            <h4>{title}</h4>
            <div class="big">{value}</div>
            <p>{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def score_card(label, score):
    st.write(label)
    st.progress(int(score) / 100)
    st.caption(f"{int(score)} / 100")



def page_home(df, models):
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(
            """
            <div class="pm-hero">
                <div class="eyebrow">University AI Course Project</div>
                <h1>PersonaMind AI 2.0</h1>
                <h3>AI Personality, Career, Behavior & Future Success Intelligence System</h3>
                <p>
                A complete student intelligence system that studies questionnaire answers, skills,
                interests, goals, habits, stress level, and writing style to generate practical AI predictions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if os.path.exists(LOGO_IMAGE):
            st.image(LOGO_IMAGE, use_container_width=True)
        elif os.path.exists(HERO_IMAGE):
            st.image(HERO_IMAGE, use_container_width=True)
        else:
            st.info("Logo or hero image will appear after the assets folder is available.")
    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Demo Records", len(df))
    with c2:
        st.metric("KNN Accuracy", f"{models['knn_accuracy']:.0%}")
    with c3:
        st.metric("Neural Net Accuracy", f"{models['mlp_accuracy']:.0%}")
    with c4:
        st.metric("Career Classes", df["career_category"].nunique())

    st.write("")
    st.markdown("### AI topics covered")
    cols = st.columns(4)
    topics = [
        ("Searching", "Career matching by skill overlap"),
        ("Agents", "Rule-based advisor combines outputs"),
        ("NLP", "Text cleaning and TF-IDF prediction"),
        ("KNN", "Career prediction classifier"),
        ("K-Means", "Behavior group clustering"),
        ("Confusion Matrix", "Model error visualization"),
        ("Neural Network", "MLPClassifier prediction model"),
    ]
    for i, (name, desc) in enumerate(topics):
        with cols[i % 4]:
            card(name, "OK", desc)



def page_assessment(models):
    st.title("User Assessment")
    intro_left, intro_right = st.columns([1.2, 1])
    with intro_left:
        st.markdown(
            """
            <div class="pm-note">
            Choose from ready-made options, add your own custom answers, then generate a full AI profile.
            The form checks missing or weak input before prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Tip: select at least two subjects, interests, and skills for better results.")
    with intro_right:
        if os.path.exists(PROFILE_IMAGE):
            st.image(PROFILE_IMAGE, use_container_width=True)

    with st.form("assessment_form"):
        st.markdown("### Basic profile")
        c1, c2, c3 = st.columns(3)
        with c1:
            name = st.text_input("Name")
            age = st.number_input("Age", min_value=1, max_value=100, value=20, step=1)
            gender = st.selectbox("Gender optional", ["Prefer not to say", "Female", "Male", "Other"])
            study_hours = st.slider("Study hours per day", 0.0, 12.0, 4.0, 0.5)
        with c2:
            selected_subjects = st.multiselect(
                "Favorite subjects",
                SUBJECT_OPTIONS,
                default=["Artificial Intelligence", "Programming", "Mathematics"],
            )
            custom_subjects = st.text_input("Add other subjects", placeholder="Example: Physics, Economics")
            selected_interests = st.multiselect(
                "Interests",
                INTEREST_OPTIONS,
                default=["Robotics", "Data analysis", "Problem solving"],
            )
            custom_interests = st.text_input("Add other interests", placeholder="Example: animation, finance")
        with c3:
            selected_skills = st.multiselect(
                "Current skills",
                SKILL_OPTIONS,
                default=["python", "communication", "research"],
            )
            custom_skills = st.text_area("Add other skills", placeholder="Example: django, flask, figma", height=78)
            career_goal = st.selectbox("Career goal", CAREER_OPTIONS, index=0)
            custom_career_goal = st.text_input("Other career goal", placeholder="Write here if not in the list")

        favorite_subjects = combine_choices(selected_subjects, custom_subjects)
        interests = combine_choices(selected_interests, custom_interests)
        current_skills = combine_choices(selected_skills, custom_skills)
        if custom_career_goal.strip():
            career_goal = custom_career_goal.strip()

        st.markdown("### Behavior and personality levels")
        c3a, c3b, c3c = st.columns(3)
        with c3a:
            stress_level = st.slider("Stress level", 1, 10, 5)
            communication = st.slider("Communication level", 1, 10, 6)
            leadership = st.slider("Leadership interest", 1, 10, 5)
        with c3b:
            creativity = st.slider("Creativity level", 1, 10, 6)
            problem_solving = st.slider("Problem-solving level", 1, 10, 7)
            risk_taking = st.slider("Risk-taking level", 1, 10, 5)
        with c3c:
            teamwork = st.slider("Teamwork preference", 1, 10, 7)
            confidence_level = st.slider("Confidence level", 1, 10, 6)
            time_management = st.slider("Time management level", 1, 10, 6)

        work_environment = st.selectbox(
            "Work environment preference",
            [
                "No preference",
                "Remote Work",
                "Team Work",
                "Research Environment",
                "Corporate Environment",
                "Startup Environment",
                "Freelance Environment",
                "Academic Environment",
                "Hybrid Work",
            ],
        )
        self_text = st.text_area(
            "Short paragraph about yourself",
            height=120,
            placeholder="Example: I enjoy building AI projects, solving coding problems, and learning new tools. I like teamwork but also need quiet time to focus.",
        )

        st.markdown("### Personality questionnaire")
        q1, q2 = st.columns(2)
        with q1:
            q_social_energy = st.radio(
                "How do you usually get energy?",
                [
                    "I enjoy groups and discussion",
                    "I prefer quiet focus",
                    "Both, depending on situation",
                    "I enjoy small focused teams",
                    "I like presenting ideas to others",
                ],
            )
            q_learning_preference = st.radio(
                "How do you learn best?",
                [
                    "Visual examples and diagrams",
                    "Reading notes and writing summaries",
                    "Practice and projects",
                    "Listening and discussion",
                    "Watching tutorials then practicing",
                ],
            )
            q_decision_style = st.radio(
                "How do you make decisions?",
                [
                    "Facts and logic first",
                    "Feelings and values first",
                    "Both logic and feelings",
                    "Fast decision after quick research",
                    "Slow decision after comparing options",
                ],
            )
        with q2:
            q_pressure_response = st.radio(
                "Under pressure, you usually:",
                [
                    "Plan and solve step by step",
                    "Ask others for ideas",
                    "Create a new approach",
                    "Take a break and restart",
                    "Focus on the most urgent task first",
                ],
            )
            q_project_role = st.radio(
                "In a project, you prefer to be:",
                [
                    "Planner or leader",
                    "Technical builder",
                    "Researcher",
                    "Creative designer",
                    "Coordinator",
                    "Presenter",
                    "Tester or debugger",
                    "Data analyst",
                ],
            )

        submitted = st.form_submit_button("Generate AI Predictions", use_container_width=True)

    if submitted:
        profile = UserProfile(
            name=name,
            age=int(age),
            gender=gender,
            favorite_subjects=favorite_subjects,
            interests=interests,
            current_skills=current_skills,
            career_goal=career_goal,
            study_hours=float(study_hours),
            stress_level=int(stress_level),
            communication=int(communication),
            leadership=int(leadership),
            creativity=int(creativity),
            problem_solving=int(problem_solving),
            risk_taking=int(risk_taking),
            teamwork=int(teamwork),
            work_environment=work_environment,
            self_text=self_text,
            q_social_energy=q_social_energy,
            q_learning_preference=q_learning_preference,
            q_decision_style=q_decision_style,
            q_pressure_response=q_pressure_response,
            q_project_role=q_project_role,
        )
        errors = validate_profile(profile)
        if errors:
            for error in errors:
                st.warning(error)
        else:
            try:
                st.session_state.profile = profile
                st.session_state.predictions = generate_predictions(profile, models)
                st.success("Predictions generated successfully. Open the AI Predictions page to view all results.")
            except Exception as exc:
                st.error(f"Prediction failed gracefully: {exc}")



def page_predictions():
    st.title("AI Predictions")
    profile = get_profile_from_state()
    predictions = get_predictions()
    if not profile or not predictions:
        st.warning("Please complete the User Assessment first.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        card("Personality Type", predictions["personality"], "Introvert, Extrovert, or Ambivert")
    with c2:
        card("Career Recommendation", predictions["career"], "Combined KNN, neural network, NLP, and search")
    with c3:
        card("Learning Style", predictions["learning_style"], "Detected from questionnaire behavior")

    st.markdown("### Scores")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        score_card("Leadership Potential", predictions["leadership_score"])
        score_card("Entrepreneurship", predictions["entrepreneurship_score"])
    with s2:
        score_card("Productivity", predictions["productivity_score"])
        score_card("Communication", predictions["communication_score"])
    with s3:
        score_card("Innovation", predictions["innovation_score"])
        score_card("Future Success Probability", predictions["success_probability"])
    with s4:
        card("Problem Solving", predictions["problem_solving_level"], "Low, Medium, or High")
        card("Burnout Risk", predictions["burnout_risk"], "Low, Medium, or High")

    st.markdown("### More Predictions")
    m1, m2, m3 = st.columns(3)
    with m1:
        card("Team Role", predictions["team_role"], f"Best teammate: {predictions['teammate']}")
        card("Hidden Talent", predictions["hidden_talent"], "Strongest detected natural ability")
    with m2:
        card("Decision Style", predictions["decision_style"], "Logical, Emotional, or Balanced")
        card("Risk Taking", predictions["risk_level"], "Predicted risk-taking level")
    with m3:
        card("Behavior Cluster", predictions["cluster_name"], "K-Means user segment")
        card("Best Work Environment", predictions["best_environment"], "Predicted fit")

    st.markdown("### AI Twin Summary")
    st.info(predictions["ai_twin"])



def page_roadmap():
    st.title("Career Roadmap")
    profile = get_profile_from_state()
    predictions = get_predictions()
    if not profile or not predictions:
        st.warning("Please complete the User Assessment first.")
        return

    career = predictions["career"]
    st.subheader(career)
    st.write(CAREER_DATABASE[career]["description"])

    required = CAREER_DATABASE[career]["skills"]
    current = skill_tokens(profile.current_skills)
    missing = [skill for skill in required if skill not in current]

    roadmap = [
        ("Month 1", "Strengthen fundamentals", f"Revise {', '.join(required[:2])}."),
        ("Month 2", "Build a mini project", f"Create one small {career.lower()} portfolio project."),
        ("Month 3", "Practice teamwork", "Join a group assignment, hackathon, or study team."),
        ("Month 4", "Close skill gaps", f"Focus on: {', '.join(missing) if missing else 'advanced practice'}."),
        ("Month 5", "Prepare portfolio", "Document projects, results, and learning reflections."),
        ("Month 6", "Career readiness", "Practice interviews, presentations, and resume storytelling."),
    ]
    for phase, title, detail in roadmap:
        st.markdown(f"**{phase}: {title}**")
        st.write(detail)



def page_skill_gap():
    st.title("Skill Gap Analysis")
    profile = get_profile_from_state()
    predictions = get_predictions()
    if not profile or not predictions:
        st.warning("Please complete the User Assessment first.")
        return
    st.dataframe(predictions["search_df"], use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=predictions["search_df"], x="Compatibility", y="Career", ax=ax, palette="viridis")
    ax.set_xlim(0, 100)
    ax.set_title("Dream Job Compatibility Score")
    st.pyplot(fig)



def page_team():
    st.title("Team Compatibility")
    predictions = get_predictions()
    if not predictions:
        st.warning("Please complete the User Assessment first.")
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        card("Your Best Team Role", predictions["team_role"], "Predicted from leadership, creativity, teamwork, and problem solving")
    with c2:
        card("Best Teammate Type", predictions["teammate"], "Balances your strongest role")
    with c3:
        card("Behavior Group", predictions["cluster_name"], "K-Means clustering result")

    st.markdown("### AI Agent Advice")
    profile = get_profile_from_state()
    for item in advisor(profile, predictions):
        st.success(item)



def page_evaluation(df, models):
    st.title("Model Evaluation")
    labels = list(models["label_encoder"].classes_)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("KNN Accuracy", f"{models['knn_accuracy']:.2%}")
    with c2:
        st.metric("Neural Network Accuracy", f"{models['mlp_accuracy']:.2%}")

    model_choice = st.selectbox("Select model for confusion matrix", ["KNN", "Neural Network"])
    pred = models["knn_pred"] if model_choice == "KNN" else models["mlp_pred"]
    cm = confusion_matrix(models["y_test"], pred)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{model_choice} Confusion Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    st.pyplot(fig)

    st.markdown("### Classification Report")
    report = classification_report(models["y_test"], pred, target_names=labels, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)



def page_about():
    st.title("About Project")
    st.write(
        """
        PersonaMind AI 2.0 is a 4th semester BS AI project that demonstrates how multiple AI techniques
        can work together in one practical student-intelligence system.
        """
    )
    st.markdown("### How each AI topic is used")
    st.write("**Searching Algorithms:** compares user skills with a career database and returns closest matches.")
    st.write("**Agents:** a rule-based AI advisor combines model outputs into career, study, skill, and wellbeing advice.")
    st.write("**NLP:** cleans user text and converts it with TF-IDF for text-based career prediction.")
    st.write("**KNN:** predicts career category from numerical student behavior features.")
    st.write("**K-Means:** clusters students into behavior groups such as analytical or creative learners.")
    st.write("**Confusion Matrix:** visualizes model prediction mistakes and correct classifications.")
    st.write("**Neural Networks:** uses scikit-learn MLPClassifier to predict career category.")

    st.markdown("### Folder structure")
    st.code(
        """PersonaMind AI 2.0/
|-- frontend.py      # frontend / Streamlit UI
|-- backend.py       # backend / AI logic
|-- requirements.txt
|-- README.md
|-- START_HERE.txt
|-- assets/
|   |-- logo.png
|   |-- persona-hero.svg
|   `-- profile-lab.svg
`-- data/
    |-- sample_persona_dataset.csv
    |-- sample_persona_dataset.xlsx
    `-- README.md""",
        language="text",
    )



def main():
    page_config()
    if os.path.exists(LOGO_IMAGE):
        st.sidebar.image(LOGO_IMAGE, use_container_width=True)
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("University AI Course Project")

    try:
        df = load_dataset()
        models = train_models(df)
    except Exception as exc:
        st.error(f"Model training failed gracefully: {exc}")
        st.stop()

    page = st.sidebar.radio(
        "Pages",
        [
            "Home",
            "User Assessment",
            "AI Predictions",
            "Career Roadmap",
            "Skill Gap Analysis",
            "Team Compatibility",
            "Model Evaluation",
            "About Project",
        ],
    )

    if page == "Home":
        page_home(df, models)
    elif page == "User Assessment":
        page_assessment(models)
    elif page == "AI Predictions":
        page_predictions()
    elif page == "Career Roadmap":
        page_roadmap()
    elif page == "Skill Gap Analysis":
        page_skill_gap()
    elif page == "Team Compatibility":
        page_team()
    elif page == "Model Evaluation":
        page_evaluation(df, models)
    elif page == "About Project":
        page_about()


if __name__ == "__main__":
    main()
