import os
import joblib
import pandas as pd
import streamlit as st


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.html("""
<style>

    /* Main page background */
    .stApp {
        background: #f5f8fc;
    }

    /* Remove default top padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #07152f 0%, #0b2145 100%);
    }

    section[data-testid="stSidebar"] * {
        color: white !important;
    }

    /* Hero section */
    .hero {
        background:
            linear-gradient(
                100deg,
                rgba(5, 25, 58, 0.96),
                rgba(18, 75, 130, 0.82)
            );
        border-radius: 24px;
        padding: 40px 45px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 12px 35px rgba(20, 55, 100, 0.18);
    }

    .hero-small {
        letter-spacing: 4px;
        font-size: 13px;
        color: #79c7ff;
        font-weight: 700;
    }

    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin: 8px 0;
    }

    .hero-subtitle {
        font-size: 17px;
        opacity: 0.9;
        max-width: 700px;
    }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        border: 1px solid #e2eaf4;
        box-shadow: 0 7px 25px rgba(26, 54, 93, 0.07);
        height: 145px;
    }

    .kpi-icon {
        font-size: 28px;
    }

    .kpi-title {
        color: #64748b;
        font-size: 13px;
        margin-top: 5px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 800;
        color: #12233f;
        margin-top: 5px;
    }

    .kpi-growth {
        color: #16a34a;
        font-size: 12px;
        font-weight: 600;
    }

    /* Section title */
    .section-title {
        font-size: 26px;
        font-weight: 800;
        color: #10233f;
        margin-top: 25px;
    }

    .section-subtitle {
        color: #718096;
        font-size: 14px;
        margin-bottom: 15px;
    }

    /* Prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #e9f4ff, #ffffff);
        border: 1px solid #d9eafa;
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 8px 28px rgba(29, 78, 130, 0.08);
    }

    .prediction-label {
        color: #718096;
        font-size: 14px;
    }

    .prediction-mode {
        color: #1677ff;
        font-size: 36px;
        font-weight: 800;
        margin-top: 5px;
    }

    .prediction-badge {
        display: inline-block;
        background: #e7f8ef;
        color: #16804b;
        padding: 6px 13px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
    }

    /* Recommendation card */
    .recommend-card {
        background: white;
        border: 1px solid #e4ebf3;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 10px;
        box-shadow: 0 6px 20px rgba(30, 55, 90, 0.06);
    }

    .recommend-rank {
        background: #1677ff;
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
    }

    .recommend-name {
        font-size: 17px;
        font-weight: 800;
        color: #13243d;
    }

    .recommend-info {
        color: #718096;
        font-size: 12px;
    }

    .score {
        color: #f59e0b;
        font-weight: 700;
    }

    /* Input area */
    .input-box {
        background: rgba(255,255,255,0.08);
        border-radius: 15px;
        padding: 12px;
        margin-bottom: 10px;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #718096;
        padding: 25px;
        font-size: 13px;
    }

</style>
""")


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

@st.cache_data
def load_data():

    # Define data files
    files = {
        "City": "data/City.xlsx",
        "Continent": "data/Continent.xlsx",
        "Country": "data/Country.xlsx",
        "Item": "data/Updated_Item.xlsx",
        "Mode": "data/Mode.xlsx",
        "Region": "data/Region.xlsx",
        "Transaction": "data/Transaction.xlsx",
        "Type": "data/Type.xlsx",
        "User": "data/User.xlsx"
    }

    # Read all Excel files
    data = {
        name: pd.read_excel(path)
        for name, path in files.items()
    }

    return data


@st.cache_resource
def load_models():

    # Load trained classification model
    model = joblib.load(
        "models/classification_model.pkl"
    )

    # Load preprocessing pipeline
    preprocessor = joblib.load(
        "models/preprocessor.pkl"
    )

    # Load attraction profile
    attraction_profile = pd.read_pickle(
        "models/attraction_profile.pkl"
    )

    # Load cosine similarity matrix
    cosine_sim = joblib.load(
        "models/cosine_similarity.pkl"
    )

    return model, preprocessor, attraction_profile, cosine_sim


data = load_data()
model, preprocessor, attraction_profile, cosine_sim = load_models()


# ---------------------------------------------------------
# CREATE ATTRACTION CATEGORY
# ---------------------------------------------------------

item = data["Item"].copy()
type_df = data["Type"].copy()


def get_attraction_category(row):

    # Get original type value
    value = row["AttractionTypeId"]

    # Return text category directly
    if isinstance(value, str):
        return value

    # Find matching type ID
    match = type_df[
        type_df["AttractionTypeId"] == value
    ]

    # Return mapped category
    if not match.empty:
        return match.iloc[0]["AttractionType"]

    # Return unknown category
    return "Unknown"


# Create category column
if "AttractionCategory" not in item.columns:

    item["AttractionCategory"] = item.apply(
        get_attraction_category,
        axis=1
    )


# ---------------------------------------------------------
# CREATE MASTER DATASET
# ---------------------------------------------------------

transaction = data["Transaction"].copy()
user = data["User"].copy()
city = data["City"].copy()
country = data["Country"].copy()
region = data["Region"].copy()
continent = data["Continent"].copy()
mode = data["Mode"].copy()


# Merge transaction with user
master = transaction.merge(
    user,
    on="UserId",
    how="left",
    validate="many_to_one"
)


# Merge attraction information
master = master.merge(
    item[
        [
            "AttractionId",
            "AttractionCityId",
            "AttractionTypeId",
            "Attraction",
            "AttractionAddress",
            "AttractionCategory"
        ]
    ],
    on="AttractionId",
    how="left",
    validate="many_to_one"
)


# Add attraction city
master = master.merge(
    city[
        ["CityId", "CityName"]
    ].rename(
        columns={
            "CityId": "AttractionCityId",
            "CityName": "AttractionCityName"
        }
    ),
    on="AttractionCityId",
    how="left",
    validate="many_to_one"
)


# Add country
master = master.merge(
    country[
        ["CountryId", "Country"]
    ].rename(
        columns={
            "Country": "CountryName"
        }
    ),
    on="CountryId",
    how="left",
    validate="many_to_one"
)


# Add region
master = master.merge(
    region[
        ["RegionId", "Region"]
    ].rename(
        columns={
            "Region": "RegionName"
        }
    ),
    on="RegionId",
    how="left",
    validate="many_to_one"
)


# Add continent
master = master.merge(
    continent[
        ["ContinentId", "Continent"]
    ].rename(
        columns={
            "Continent": "ContinentName"
        }
    ),
    on="ContinentId",
    how="left",
    validate="many_to_one"
)


# Add readable visit mode
master = master.merge(
    mode[
        ["VisitModeId", "VisitMode"]
    ].rename(
        columns={
            "VisitMode": "VisitModeName"
        }
    ),
    left_on="VisitMode",
    right_on="VisitModeId",
    how="left",
    validate="many_to_one"
)


# Remove duplicate mode ID
master = master.drop(
    columns=["VisitModeId"]
)


# Add user city
master = master.merge(
    city[
        ["CityId", "CityName"]
    ].rename(
        columns={
            "CityName": "UserCityName"
        }
    ),
    on="CityId",
    how="left",
    validate="many_to_one"
)


# Fill missing city
master["UserCityName"] = master[
    "UserCityName"
].fillna("Unknown")


# Fill missing region
master["RegionName"] = master[
    "RegionName"
].replace("-", "Unknown")


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------

def get_season(month):

    # Return season from month
    if month in [12, 1, 2]:
        return "Winter"

    elif month in [3, 4, 5]:
        return "Spring"

    elif month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


# Create season
master["Season"] = master[
    "VisitMonth"
].apply(get_season)


# Count user transactions
user_counts = master[
    "UserId"
].value_counts()

master["UserTransactionCount"] = master[
    "UserId"
].map(user_counts)


# Count attraction popularity
attraction_counts = master[
    "AttractionId"
].value_counts()

master["AttractionPopularity"] = master[
    "AttractionId"
].map(attraction_counts)


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.html("""
    <div style="text-align:center; padding:10px;">
        <div style="font-size:45px;">🌍</div>
        <div style="font-size:23px;font-weight:800;">
            Tourism
        </div>
        <div style="font-size:15px;color:#65b8ff;">
            Experience Analytics
        </div>
    </div>
    """)

    st.markdown("---")

    st.markdown("### 🏠 Navigation")

    page = st.radio(
        "Go to",
        [
            "Dashboard",
            "Prediction",
            "Recommendations",
            "Insights"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### 👤 User Input")

    # Get valid user IDs
    valid_user_ids = sorted(
        user["UserId"].dropna().unique()
    )

    # Use first valid user as default
    default_user = int(
        valid_user_ids[0]
    )

    user_id = st.number_input(
        "User ID",
        min_value=int(min(valid_user_ids)),
        max_value=int(max(valid_user_ids)),
        value=default_user,
        step=1
    )

    visit_year = st.selectbox(
        "Visit Year",
        sorted(
            master["VisitYear"].unique(),
            reverse=True
        )
    )

    visit_month = st.selectbox(
        "Visit Month",
        list(range(1, 13))
    )

    st.markdown("")

    st.info(
        "💡 Select a User ID to explore "
        "personalized tourism insights."
    )


# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------

st.html("""
<div class="hero">

    <div class="hero-small">
        WELCOME TO
    </div>

    <div class="hero-title">
        🌍 Tourism Experience Analytics
    </div>

    <div class="hero-subtitle">
        Classification, Prediction and Recommendation System
        <br><br>
        Explore visitor behaviour, predict visit mode and
        discover personalized attractions.
    </div>

</div>
""")


# ---------------------------------------------------------
# KPI CARDS
# ---------------------------------------------------------

total_transactions = len(transaction)
total_users = user["UserId"].nunique()
total_attractions = item["AttractionId"].nunique()
average_rating = master["Rating"].mean()


c1, c2, c3, c4 = st.columns(4)


with c1:

    st.html(f"""
    <div class="kpi-card">

        <div class="kpi-icon">🗃️</div>

        <div class="kpi-title">
            Total Transactions
        </div>

        <div class="kpi-value">
            {total_transactions:,}
        </div>

        <div class="kpi-growth">
            ↑ Tourism activity
        </div>

    </div>
    """)


with c2:

    st.html(f"""
    <div class="kpi-card">

        <div class="kpi-icon">👥</div>

        <div class="kpi-title">
            Total Users
        </div>

        <div class="kpi-value">
            {total_users:,}
        </div>

        <div class="kpi-growth">
            ↑ Active visitors
        </div>

    </div>
    """)


with c3:

    st.html(f"""
    <div class="kpi-card">

        <div class="kpi-icon">📍</div>

        <div class="kpi-title">
            Attractions
        </div>

        <div class="kpi-value">
            {total_attractions:,}
        </div>

        <div class="kpi-growth">
            ↑ Available attractions
        </div>

    </div>
    """)


with c4:

    st.html(f"""
    <div class="kpi-card">

        <div class="kpi-icon">⭐</div>

        <div class="kpi-title">
            Average Rating
        </div>

        <div class="kpi-value">
            {average_rating:.2f}
        </div>

        <div class="kpi-growth">
            ↑ Overall experience
        </div>

    </div>
    """)


# ---------------------------------------------------------
# DASHBOARD PAGE
# ---------------------------------------------------------

if page == "Dashboard":

    st.html(
        '<div class="section-title">✨ Explore Your Journey</div>'
    )

    st.html(
        '<div class="section-subtitle">'
        'Use historical tourism data to understand visitor behaviour.'
        '</div>'
    )

    left, right = st.columns([1, 1])


    # -----------------------------------------------------
    # VISIT MODE PREDICTION
    # -----------------------------------------------------

    with left:

        st.html("""
        <div class="prediction-card">

            <div style="font-size:30px;">
                🧠
            </div>

            <div style="font-size:22px;font-weight:800;color:#12233f;">
                Visit Mode Prediction
            </div>

            <div class="prediction-label">
                Based on visitor profile and historical data
            </div>

        </div>
        """)

        st.markdown("")

        # Select attraction
        attraction_names = sorted(
            master["Attraction"]
            .dropna()
            .unique()
        )

        selected_attraction = st.selectbox(
            "Select Attraction",
            attraction_names
        )

        if st.button(
            "🔮 Predict Visit Mode",
            use_container_width=True
        ):

            # Get user history
            selected_user = master[
                master["UserId"] == user_id
            ]

            if selected_user.empty:

                st.warning(
                    "User ID not found in the dataset."
                )

            else:

                # Get selected attraction row
                attraction_rows = master[
                    master["Attraction"] ==
                    selected_attraction
                ]

                attraction_row = (
                    attraction_rows.iloc[0]
                )

                user_row = (
                    selected_user.iloc[0]
                )

                # Build model input
                prediction_data = pd.DataFrame([{

                    "VisitYear": visit_year,

                    "VisitMonth": visit_month,

                    "Season": get_season(
                        visit_month
                    ),

                    "AttractionCategory":
                        attraction_row[
                            "AttractionCategory"
                        ],

                    "AttractionCityName":
                        attraction_row[
                            "AttractionCityName"
                        ],

                    "CountryName":
                        attraction_row[
                            "CountryName"
                        ],

                    "RegionName":
                        attraction_row[
                            "RegionName"
                        ],

                    "ContinentName":
                        attraction_row[
                            "ContinentName"
                        ],

                    "UserCityName":
                        user_row[
                            "UserCityName"
                        ],

                    "UserTransactionCount":
                        int(
                            user_counts.get(
                                user_id,
                                0
                            )
                        ),

                    "AttractionPopularity":
                        int(
                            attraction_counts.get(
                                attraction_row[
                                    "AttractionId"
                                ],
                                0
                            )
                        )
                }])


                # Transform input
                X_input = preprocessor.transform(
                    prediction_data
                )


                # Make prediction
                prediction = model.predict(
                    X_input
                )[0]


                # Show prediction
                st.html(f"""
                <div class="prediction-card">

                    <div class="prediction-label">
                        Predicted Visit Mode
                    </div>

                    <div class="prediction-mode">
                        {prediction}
                    </div>

                    <span class="prediction-badge">
                        ❤️ Most Likely
                    </span>

                </div>
                """)


    # -----------------------------------------------------
    # QUICK RECOMMENDATIONS
    # -----------------------------------------------------

    with right:

        st.html("""
        <div class="prediction-card">

            <div style="font-size:30px;">
                ⭐
            </div>

            <div style="font-size:22px;font-weight:800;color:#12233f;">
                Personalized Recommendations
            </div>

            <div class="prediction-label">
                Top attractions based on your history
            </div>

        </div>
        """)

        st.markdown("")

        if st.button(
            "⭐ Get Recommendations",
            use_container_width=True
        ):

            user_history = master[
                master["UserId"] == user_id
            ]

            if user_history.empty:

                st.warning(
                    "User ID not found in the dataset."
                )

            else:

                visited_ids = (
                    user_history[
                        "AttractionId"
                    ].unique()
                )

                user_ratings = (
                    user_history
                    .groupby("AttractionId")[
                        "Rating"
                    ]
                    .mean()
                )

                attraction_index_map = pd.Series(
                    attraction_profile.index,
                    index=attraction_profile[
                        "AttractionId"
                    ]
                )

                recommendations = []

                # Score each attraction
                for _, attraction in (
                    attraction_profile.iterrows()
                ):

                    attraction_id = (
                        attraction["AttractionId"]
                    )

                    if attraction_id in visited_ids:
                        continue

                    candidate_idx = (
                        attraction_index_map[
                            attraction_id
                        ]
                    )

                    similarities = []

                    for visited_id, rating in (
                        user_ratings.items()
                    ):

                        if visited_id not in (
                            attraction_index_map.index
                        ):
                            continue

                        visited_idx = (
                            attraction_index_map[
                                visited_id
                            ]
                        )

                        similarity = cosine_sim[
                            candidate_idx,
                            visited_idx
                        ]

                        similarities.append(
                            similarity *
                            (rating / 5)
                        )

                    if similarities:

                        user_similarity = (
                            sum(similarities)
                            / len(similarities)
                        )

                    else:

                        user_similarity = 0


                    # Calculate final recommendation score
                    final_score = (
                        0.70 * user_similarity
                        +
                        0.25 *
                        attraction[
                            "RatingScore"
                        ]
                        +
                        0.05 *
                        attraction[
                            "PopularityScore"
                        ]
                    )

                    recommendations.append({

                        "Attraction":
                            attraction[
                                "Attraction"
                            ],

                        "Category":
                            attraction[
                                "AttractionCategory"
                            ],

                        "City":
                            attraction[
                                "AttractionCityName"
                            ],

                        "Rating":
                            attraction[
                                "AverageRating"
                            ],

                        "Score":
                            final_score
                    })


                # Sort recommendations
                recommendations = sorted(
                    recommendations,
                    key=lambda x: x["Score"],
                    reverse=True
                )[:5]


                # Display recommendations
                for rank, rec in enumerate(
                    recommendations,
                    start=1
                ):

                    st.html(f"""
                    <div class="recommend-card">

                        <span class="recommend-rank">
                            {rank}
                        </span>

                        &nbsp;

                        <span class="recommend-name">
                            {rec["Attraction"]}
                        </span>

                        <br>

                        <span class="recommend-info">
                            📍 {rec["City"]}
                            &nbsp; • &nbsp;
                            🏷️ {rec["Category"]}
                        </span>

                        <br>

                        <span class="score">
                            ⭐ {rec["Rating"]:.2f}
                        </span>

                        &nbsp;

                        <span class="recommend-info">
                            Recommendation Score:
                            {rec["Score"]:.2f}
                        </span>

                    </div>
                    """)


# ---------------------------------------------------------
# PREDICTION PAGE
# ---------------------------------------------------------

elif page == "Prediction":

    st.html(
        '<div class="section-title">🧠 Visit Mode Prediction</div>'
    )

    st.write(
        "Predict whether the visitor is likely to travel "
        "as Couples, Family, Friends, Solo or Business."
    )

    prediction_attraction = st.selectbox(
        "Choose Attraction",
        sorted(
            master["Attraction"]
            .dropna()
            .unique()
        )
    )

    if st.button(
        "🚀 Run Prediction",
        use_container_width=True
    ):

        selected_user = master[
            master["UserId"] == user_id
        ]

        if selected_user.empty:

            st.error(
                "User ID not found."
            )

        else:

            attraction_row = master[
                master["Attraction"] ==
                prediction_attraction
            ].iloc[0]

            user_row = selected_user.iloc[0]


            # Prepare prediction features
            prediction_data = pd.DataFrame([{

                "VisitYear": visit_year,

                "VisitMonth": visit_month,

                "Season":
                    get_season(
                        visit_month
                    ),

                "AttractionCategory":
                    attraction_row[
                        "AttractionCategory"
                    ],

                "AttractionCityName":
                    attraction_row[
                        "AttractionCityName"
                    ],

                "CountryName":
                    attraction_row[
                        "CountryName"
                    ],

                "RegionName":
                    attraction_row[
                        "RegionName"
                    ],

                "ContinentName":
                    attraction_row[
                        "ContinentName"
                    ],

                "UserCityName":
                    user_row[
                        "UserCityName"
                    ],

                "UserTransactionCount":
                    int(
                        user_counts.get(
                            user_id,
                            0
                        )
                    ),

                "AttractionPopularity":
                    int(
                        attraction_counts.get(
                            attraction_row[
                                "AttractionId"
                            ],
                            0
                        )
                    )
            }])


            # Transform input
            X_input = preprocessor.transform(
                prediction_data
            )

            # Predict class
            predicted_mode = model.predict(
                X_input
            )[0]


            st.success(
                f"🎯 Predicted Visit Mode: "
                f"**{predicted_mode}**"
            )


# ---------------------------------------------------------
# RECOMMENDATIONS PAGE
# ---------------------------------------------------------

elif page == "Recommendations":

    st.html(
        '<div class="section-title">⭐ Personalized Recommendations</div>'
    )

    st.html(
        '<div class="section-subtitle">'
        'Discover attractions based on your historical preferences.'
        '</div>'
    )


    user_history = master[
        master["UserId"] == user_id
    ]


    if user_history.empty:

        st.warning(
            "Please enter a valid User ID."
        )

    else:

        visited_ids = user_history[
            "AttractionId"
        ].unique()

        user_ratings = (
            user_history
            .groupby("AttractionId")[
                "Rating"
            ]
            .mean()
        )

        attraction_index_map = pd.Series(
            attraction_profile.index,
            index=attraction_profile[
                "AttractionId"
            ]
        )

        results = []

        for _, attraction in (
            attraction_profile.iterrows()
        ):

            attraction_id = (
                attraction["AttractionId"]
            )

            if attraction_id in visited_ids:
                continue

            candidate_idx = (
                attraction_index_map[
                    attraction_id
                ]
            )

            similarities = []

            for visited_id, rating in (
                user_ratings.items()
            ):

                if visited_id not in (
                    attraction_index_map.index
                ):
                    continue

                visited_idx = (
                    attraction_index_map[
                        visited_id
                    ]
                )

                similarity = cosine_sim[
                    candidate_idx,
                    visited_idx
                ]

                similarities.append(
                    similarity *
                    (rating / 5)
                )

            user_similarity = (
                sum(similarities)
                / len(similarities)
                if similarities
                else 0
            )


            # Calculate recommendation score
            score = (
                0.70 * user_similarity
                +
                0.25 *
                attraction["RatingScore"]
                +
                0.05 *
                attraction["PopularityScore"]
            )


            results.append({

                "Attraction":
                    attraction["Attraction"],

                "Category":
                    attraction[
                        "AttractionCategory"
                    ],

                "City":
                    attraction[
                        "AttractionCityName"
                    ],

                "Country":
                    attraction[
                        "CountryName"
                    ],

                "Average Rating":
                    attraction[
                        "AverageRating"
                    ],

                "Popularity":
                    attraction[
                        "AttractionPopularity"
                    ],

                "Recommendation Score":
                    score
            })


        results_df = pd.DataFrame(
            results
        ).sort_values(
            "Recommendation Score",
            ascending=False
        ).head(10)


        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


# ---------------------------------------------------------
# INSIGHTS PAGE
# ---------------------------------------------------------

elif page == "Insights":

    st.html(
        '<div class="section-title">📊 Tourism Insights</div>'
    )

    st.html(
        '<div class="section-subtitle">'
        'Explore important patterns from the tourism dataset.'
        '</div>'
    )


    col1, col2 = st.columns(2)


    # -----------------------------------------------------
    # VISIT MODE DISTRIBUTION
    # -----------------------------------------------------

    with col1:

        st.subheader(
            "🥧 Visit Mode Distribution"
        )

        mode_counts = (
            master["VisitModeName"]
            .value_counts()
        )

        st.bar_chart(
            mode_counts
        )


    # -----------------------------------------------------
    # RATING DISTRIBUTION
    # -----------------------------------------------------

    with col2:

        st.subheader(
            "⭐ Rating Distribution"
        )

        rating_counts = (
            master["Rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_counts
        )


    col3, col4 = st.columns(2)


    # -----------------------------------------------------
    # YEARLY VISITS
    # -----------------------------------------------------

    with col3:

        st.subheader(
            "📅 Visits by Year"
        )

        yearly_visits = (
            master["VisitYear"]
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            yearly_visits
        )


    # -----------------------------------------------------
    # MONTHLY VISITS
    # -----------------------------------------------------

    with col4:

        st.subheader(
            "🌤️ Visits by Month"
        )

        monthly_visits = (
            master["VisitMonth"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            monthly_visits
        )


    # -----------------------------------------------------
    # TOP ATTRACTIONS
    # -----------------------------------------------------

    st.subheader(
        "🔥 Most Popular Attractions"
    )

    top_attractions = (
        master["Attraction"]
        .value_counts()
        .head(10)
    )

    st.bar_chart(
        top_attractions
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.html("""
<div class="footer">

    🌍 <b>Tourism Experience Analytics</b>
    <br>
    Explore • Discover • Experience
    <br><br>
    Classification • Prediction • Recommendation

</div>
""")