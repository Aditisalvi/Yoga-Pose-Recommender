import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, ndcg_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


class YogaRecommenderSystem:
    """
    Advanced Personalized Yoga Recommender System with Neural Network Architecture
    Prioritizes safety, personalization, and effectiveness
    """

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Initialize components
        self.scaler = StandardScaler()
        self.mlb_focus = MultiLabelBinarizer()
        self.mlb_precautions = MultiLabelBinarizer()
        self.mlb_pain_points = MultiLabelBinarizer()
        self.mlb_body_parts = MultiLabelBinarizer()

        # Safety mappings - highest priority
        self.safety_conditions = {
            'high_blood_pressure': ['High blood pressure'],
            'low_blood_pressure': ['Low blood pressure'],
            'back_injuries': ['Back injuries'],
            'knee_problems': ['Knee problems/injuries'],
            'neck_injuries': ['Neck injuries'],
            'shoulder_injuries': ['Shoulder injuries'],
            'ankle_injuries': ['Ankle injuries'],
            'wrist_injuries': ['Wrist injuries'],
            'hip_injuries': ['Hip injuries'],
            'heart_conditions': ['Heart conditions'],
            'pregnancy': ['Pregnancy'],
            'glaucoma': ['Glaucoma/eye conditions'],
            'carpal_tunnel': ['Carpal tunnel'],
            'hamstring_injuries': ['Hamstring injuries'],
            'asthma': ['Asthma'],
            'migraine': ['Migraine'],
            'insomnia': ['Insomnia'],
            'diarrhea': ['Diarrhea'],
            'menstruation': ['Menstruation'],
            'balance_disorders': ['Balance disorders']
        }

        # Focus areas for recommendations
        self.focus_areas = [
            'Flexibility/Stretching', 'Strength Building', 'Balance/Stability',
            'Stress Relief/Calming', 'Posture Improvement', 'Meditation/Focus',
            'Digestion Support', 'Cardiovascular Fitness', 'Endurance Building',
            'Energy Building', 'Circulation Enhancement', 'Coordination',
            'Relaxation/Restorative', 'Emotional Release', 'Detoxification',
            'Breathing Improvement'
        ]

        # Define focus area similarity mapping
        self.focus_similarity = {
            'Flexibility/Stretching': ['Posture Improvement', 'Relaxation/Restorative'],
            'Strength Building': ['Endurance Building', 'Cardiovascular Fitness'],
            'Balance/Stability': ['Coordination', 'Posture Improvement'],
            'Stress Relief/Calming': ['Meditation/Focus', 'Relaxation/Restorative', 'Emotional Release'],
            'Posture Improvement': ['Flexibility/Stretching', 'Balance/Stability'],
            'Meditation/Focus': ['Stress Relief/Calming', 'Relaxation/Restorative'],
            'Digestion Support': ['Detoxification', 'Circulation Enhancement'],
            'Cardiovascular Fitness': ['Strength Building', 'Endurance Building'],
            'Endurance Building': ['Strength Building', 'Cardiovascular Fitness'],
            'Energy Building': ['Circulation Enhancement', 'Cardiovascular Fitness'],
            'Circulation Enhancement': ['Energy Building', 'Cardiovascular Fitness'],
            'Coordination': ['Balance/Stability', 'Posture Improvement'],
            'Relaxation/Restorative': ['Stress Relief/Calming', 'Meditation/Focus'],
            'Emotional Release': ['Stress Relief/Calming', 'Meditation/Focus'],
            'Detoxification': ['Digestion Support', 'Circulation Enhancement'],
            'Breathing Improvement': ['Stress Relief/Calming', 'Meditation/Focus']
        }

        # Fitness to yoga difficulty mapping
        self.fitness_to_yoga_mapping = {
            'A': 'Advanced',
            'B': 'Intermediate',
            'C': 'Beginner',
            'D': 'Beginner'
        }

        print("🧘 Advanced Yoga Recommender System Initialized")
        print("✅ Safety-first approach enabled")
        print("✅ Multi-modal recommendation framework ready")

    def load_and_preprocess_data(self):
        """Load and preprocess both datasets"""
        print("\n📊 Loading and preprocessing datasets...")

        try:
            # Load datasets
            self.users_df = pd.read_csv('C:/Users/salvi/PycharmProjects/yoga_recommender/bodyPerformance.csv')
            self.asanas_df = pd.read_csv('C:/Users/salvi/PycharmProjects/yoga_recommender/new_final_dataset (1).csv')

            print(f"✅ Users dataset loaded: {self.users_df.shape}")
            print(f"✅ Asanas dataset loaded: {self.asanas_df.shape}")
            print("Meow Meow")

        except FileNotFoundError:
            print("⚠️ Kaggle datasets not found, creating synthetic data for testing...")
            self._create_synthetic_data()

        # Preprocess users data
        self._preprocess_users_data()

        # Preprocess asanas data
        self._preprocess_asanas_data()

        print("✅ Data preprocessing completed")

    def _create_synthetic_data(self):
        """Create synthetic data for testing purposes"""
        np.random.seed(42)

        # Create synthetic users data
        n_users = 1000
        self.users_df = pd.DataFrame({
            'age': np.random.randint(20, 65, n_users),
            'gender': np.random.choice(['F', 'M'], n_users),
            'height_cm': np.random.normal(170, 10, n_users),
            'weight_kg': np.random.normal(70, 15, n_users),
            'body_fat_%': np.random.normal(20, 8, n_users),
            'diastolic': np.random.normal(80, 10, n_users),
            'systolic': np.random.normal(120, 15, n_users),
            'gripForce': np.random.normal(35, 10, n_users),
            'sit_and_bend_forward_cm': np.random.normal(15, 8, n_users),
            'sit_ups_counts': np.random.randint(10, 50, n_users),
            'broad_jump_cm': np.random.normal(180, 30, n_users),
            'class': np.random.choice(['A', 'B', 'C', 'D'], n_users)
        })

        # Create synthetic asanas data
        poses = ['Mountain Pose', 'Tree Pose', 'Warrior I', 'Warrior II', 'Downward Dog',
                 'Child Pose', 'Cobra Pose', 'Bridge Pose', 'Triangle Pose', 'Plank Pose',
                 'Cat-Cow', 'Pigeon Pose', 'Camel Pose', 'Fish Pose', 'Lotus Pose']

        n_asanas = 50
        self.asanas_df = pd.DataFrame({
            'asana_name': [f'{poses[i % len(poses)]}_{i}' for i in range(n_asanas)],
            'difficulty_level': np.random.choice(['Beginner', 'Intermediate', 'Advanced'], n_asanas),
            'Focus_Area': np.random.choice(self.focus_areas, n_asanas),
            'Body_Parts': np.random.choice(['Back/Spine', 'Legs/Thighs', 'Core/Abdomen', 'Arms/Wrists'], n_asanas),
            'Precautions': np.random.choice(
                ['Generally safe for all', 'Back injuries', 'Knee problems/injuries', 'High blood pressure'], n_asanas),
            'Pain_Points': np.random.choice(['Lower back pain', 'Stress/anxiety', 'Balance issues', 'Hip tightness'],
                                            n_asanas),
            'duration_secs': np.random.randint(30, 180, n_asanas),
            'target_age_group': ['8-80'] * n_asanas
        })

    def _preprocess_users_data(self):
        """Preprocess users dataset with comprehensive feature engineering"""
        print("   Processing users data...")

        # Clean column names
        self.users_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for
                                 col in self.users_df.columns]

        # Handle missing values
        numeric_cols = self.users_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.users_df[col].fillna(self.users_df[col].median(), inplace=True)

        # Create comprehensive health and fitness features
        self.users_df['bmi'] = self.users_df['weight_kg'] / (self.users_df['height_cm'] / 100) ** 2

        # Blood pressure risk assessment
        self.users_df['bp_systolic'] = self.users_df.get('systolic', 120)
        self.users_df['bp_diastolic'] = self.users_df.get('diastolic', 80)

        self.users_df['bp_risk'] = np.where(
            (self.users_df['bp_systolic'] > 140) | (self.users_df['bp_diastolic'] > 90),
            'high_bp',
            np.where(
                (self.users_df['bp_systolic'] < 90) | (self.users_df['bp_diastolic'] < 60),
                'low_bp', 'normal_bp'
            )
        )

        # Physical capability scores (0-100 scale)
        self.users_df['flexibility_score'] = np.clip(
            (self.users_df.get('sit_and_bend_forward_cm', 15) + 10) * 2.5, 0, 100
        )

        self.users_df['strength_score'] = np.clip(
            self.users_df.get('gripforce', 35) * 2, 0, 100
        )

        self.users_df['balance_score'] = np.clip(
            self.users_df.get('broad_jump_cm', 180) / 3, 0, 100
        )

        self.users_df['cardio_score'] = np.clip(
            100 - self.users_df.get('body_fat_%', 20), 0, 100
        )

        # Fitness level mapping
        self.users_df['yoga_difficulty_level'] = self.users_df['class'].map(self.fitness_to_yoga_mapping)

        # Overall fitness composite score
        self.users_df['fitness_composite'] = (
                self.users_df['flexibility_score'] * 0.25 +
                self.users_df['strength_score'] * 0.25 +
                self.users_df['balance_score'] * 0.25 +
                self.users_df['cardio_score'] * 0.25
        )

        # Age group categorization
        self.users_df['age_group'] = pd.cut(
            self.users_df['age'],
            bins=[0, 30, 45, 60, 100],
            labels=['Young', 'Adult', 'Middle_Age', 'Senior']
        )

        # Risk assessment
        self.users_df['practice_risk_level'] = 'low'

        high_risk = (
                (self.users_df['bp_risk'] == 'high_bp') |
                (self.users_df['age'] > 65) |
                (self.users_df['bmi'] > 35) |
                (self.users_df['fitness_composite'] < 25)
        )
        self.users_df.loc[high_risk, 'practice_risk_level'] = 'high'

        medium_risk = (
                (self.users_df['bp_risk'] == 'low_bp') |
                (self.users_df['age'] > 55) |
                (self.users_df['bmi'] > 30) |
                (self.users_df['fitness_composite'] < 40)
        )
        self.users_df.loc[
            medium_risk & (self.users_df['practice_risk_level'] == 'low'), 'practice_risk_level'] = 'medium'

        print(f"   ✅ Users processed: {self.users_df.shape}")

    def _preprocess_asanas_data(self):
        """Preprocess asanas dataset with advanced feature engineering"""
        print("   Processing asanas data...")

        # Clean column names
        self.asanas_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for
                                  col in self.asanas_df.columns]

        # Handle missing values
        self.asanas_df['asana_name'].fillna('Unknown_Pose', inplace=True)
        self.asanas_df['difficulty_level'].fillna('Beginner', inplace=True)

        # Parse multi-value fields
        def parse_multi_value(field_value):
            if pd.isna(field_value):
                return []
            items = str(field_value).split(',')
            return [item.strip().replace('[', '').replace(']', '').replace("'", "") for item in items if item.strip()]

        self.asanas_df['focus_area_list'] = self.asanas_df.get('focus_area', 'Flexibility/Stretching').apply(
            parse_multi_value)
        self.asanas_df['body_parts_list'] = self.asanas_df.get('body_parts', 'Full Body').apply(parse_multi_value)
        self.asanas_df['precautions_list'] = self.asanas_df.get('precautions', 'Generally safe for all').apply(
            parse_multi_value)
        self.asanas_df['pain_points_list'] = self.asanas_df.get('pain_points', 'General wellness').apply(
            parse_multi_value)

        # Create difficulty scores
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.asanas_df['difficulty_score'] = self.asanas_df['difficulty_level'].map(difficulty_mapping)

        # Create complexity score
        self.asanas_df['complexity_score'] = self.asanas_df['difficulty_score'].copy()

        # Create binary feature matrices
        focus_matrix = self.mlb_focus.fit_transform(self.asanas_df['focus_area_list'])
        focus_columns = [f'focus_{area.lower().replace("/", "_").replace(" ", "_")}'
                         for area in self.mlb_focus.classes_]
        focus_df = pd.DataFrame(focus_matrix, columns=focus_columns, index=self.asanas_df.index)

        body_matrix = self.mlb_body_parts.fit_transform(self.asanas_df['body_parts_list'])
        body_columns = [f'body_{part.lower().replace("/", "_").replace(" ", "_")}'
                        for part in self.mlb_body_parts.classes_]
        body_df = pd.DataFrame(body_matrix, columns=body_columns, index=self.asanas_df.index)

        precautions_matrix = self.mlb_precautions.fit_transform(self.asanas_df['precautions_list'])
        precautions_columns = [f'precaution_{prec.lower().replace("/", "_").replace(" ", "_")}'
                               for prec in self.mlb_precautions.classes_]
        precautions_df = pd.DataFrame(precautions_matrix, columns=precautions_columns, index=self.asanas_df.index)

        pain_matrix = self.mlb_pain_points.fit_transform(self.asanas_df['pain_points_list'])
        pain_columns = [f'pain_{point.lower().replace("/", "_").replace(" ", "_")}'
                        for point in self.mlb_pain_points.classes_]
        pain_df = pd.DataFrame(pain_matrix, columns=pain_columns, index=self.asanas_df.index)

        # Combine all features
        self.asanas_processed = pd.concat([
            self.asanas_df[['asana_name', 'difficulty_level', 'difficulty_score', 'complexity_score']],
            focus_df,
            body_df,
            precautions_df,
            pain_df
        ], axis=1)

        # Create safety score
        self.asanas_processed['safety_score'] = 1.0

        # Reduce safety for risky poses
        risky_precautions = ['high_blood_pressure', 'heart_conditions', 'back_injuries',
                             'knee_problems_injuries', 'neck_injuries', 'pregnancy']
        for precaution in risky_precautions:
            precaution_col = f'precaution_{precaution}'
            if precaution_col in self.asanas_processed.columns:
                self.asanas_processed.loc[self.asanas_processed[precaution_col] == 1, 'safety_score'] -= 0.2

        # Modified accessibility score
        self.asanas_processed['accessibility_score'] = 1.0

        focus_columns = [col for col in self.asanas_processed.columns if col.startswith('focus_')]
        self.asanas_processed['effectiveness_score'] = self.asanas_processed[focus_columns].sum(axis=1) / max(
            len(focus_columns), 1)

        # Store feature columns
        self.focus_feature_columns = focus_columns
        self.body_feature_columns = [col for col in self.asanas_processed.columns if col.startswith('body_')]
        self.precaution_columns = [col for col in self.asanas_processed.columns if col.startswith('precaution_')]
        self.pain_columns = [col for col in self.asanas_processed.columns if col.startswith('pain_')]

        print(f"   ✅ Asanas processed: {self.asanas_processed.shape}")
        print(f"   ✅ Focus features: {len(self.focus_feature_columns)}")
        print(f"   ✅ Safety features: {len(self.precaution_columns)}")

    def _process_user_input(self, user_input):
        """Process and validate user input for recommendations"""
        user_profile = {}

        # Basic demographics
        user_profile['age'] = user_input.get('age', 35)
        user_profile['weight'] = user_input.get('weight', 70)
        user_profile['height'] = user_input.get('height', 170)
        user_profile['gender'] = user_input.get('gender', 'M')

        # Calculate BMI
        user_profile['bmi'] = user_profile['weight'] / (user_profile['height'] / 100) ** 2

        # Physical and yoga levels
        user_profile['physical_level'] = user_input.get('physical_level', 'Beginner')
        user_profile['yoga_experience'] = user_input.get('yoga_experience', 'Beginner')

        # Focus areas
        user_profile['focus_area'] = user_input.get('focus_area', 'Flexibility/Stretching')

        # Health conditions and precautions
        user_profile['health_conditions'] = user_input.get('health_conditions', [])
        user_profile['injuries'] = user_input.get('injuries', [])

        # Blood pressure (if provided)
        user_profile['bp_systolic'] = user_input.get('bp_systolic', 120)
        user_profile['bp_diastolic'] = user_input.get('bp_diastolic', 80)

        # Determine BP risk
        if user_profile['bp_systolic'] > 140 or user_profile['bp_diastolic'] > 90:
            user_profile['bp_risk'] = 'high_bp'
        elif user_profile['bp_systolic'] < 90 or user_profile['bp_diastolic'] < 60:
            user_profile['bp_risk'] = 'low_bp'
        else:
            user_profile['bp_risk'] = 'normal_bp'

        # Estimated fitness scores based on age and BMI
        age_factor = max(0, 1 - (user_profile['age'] - 25) / 50)
        bmi_factor = max(0, 1 - abs(user_profile['bmi'] - 22) / 10)

        user_profile['flexibility_score'] = 50 + age_factor * 30 + bmi_factor * 20
        user_profile['strength_score'] = 50 + age_factor * 25 + bmi_factor * 25
        user_profile['balance_score'] = 50 + age_factor * 30 + bmi_factor * 20
        user_profile['cardio_score'] = 50 + age_factor * 25 + bmi_factor * 25

        user_profile['fitness_composite'] = (
                user_profile['flexibility_score'] * 0.25 +
                user_profile['strength_score'] * 0.25 +
                user_profile['balance_score'] * 0.25 +
                user_profile['cardio_score'] * 0.25
        )

        # Map physical level to difficulty score
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        user_profile['difficulty_score'] = difficulty_mapping.get(user_profile['physical_level'], 1)

        return user_profile

    def _filter_safe_poses(self, user_profile):
        """Filter poses based on safety conditions (highest priority)"""
        safe_poses = self.asanas_processed.copy()

        # Check health conditions and injuries
        all_conditions = user_profile.get('health_conditions', []) + user_profile.get('injuries', [])

        for condition in all_conditions:
            condition_normalized = condition.lower().replace(' ', '_').replace('/', '_')
            precaution_col = f'precaution_{condition_normalized}'

            if precaution_col in safe_poses.columns:
                safe_poses = safe_poses[safe_poses[precaution_col] == 0]

        # Blood pressure specific filtering
        if user_profile.get('bp_risk') == 'high_bp':
            if 'precaution_high_blood_pressure' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_high_blood_pressure'] == 0]

        if user_profile.get('bp_risk') == 'low_bp':
            if 'precaution_low_blood_pressure' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_low_blood_pressure'] == 0]

        # Age-based filtering
        if user_profile.get('age', 35) > 65:
            safe_poses = safe_poses[safe_poses['complexity_score'] <= 2.0]

        # BMI-based filtering
        if user_profile.get('bmi', 25) > 35:
            if 'precaution_back_injuries' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_back_injuries'] == 0]

        print(f"   Safety filter: {len(safe_poses)} poses remaining from {len(self.asanas_processed)}")
        return safe_poses

    def _filter_by_focus_area(self, poses_df, user_profile):
        """Filter poses by user's focus area with robust matching."""
        focus_area = user_profile.get('focus_area', 'Flexibility/Stretching')
        focus_normalized = focus_area.lower().replace('/', '_').replace(' ', '_')
        focus_col = f'focus_{focus_normalized}'
        print(f"   Filtering for focus area: {focus_area} (normalized: {focus_col})")
        if focus_col in poses_df.columns:
            direct_matches = poses_df[poses_df[focus_col] == 1]
            if len(direct_matches) > 0:
                beginner_count = len(direct_matches[direct_matches['difficulty_score'] == 1])
                intermediate_count = len(direct_matches[direct_matches['difficulty_score'] == 2])
                advanced_count = len(direct_matches[direct_matches['difficulty_score'] == 3])
                print(f"   Focus area filter: {len(direct_matches)} direct matches found "
                      f"({beginner_count} Beginner, {intermediate_count} Intermediate, {advanced_count} Advanced)")
                return direct_matches
        similar_areas = self.focus_similarity.get(focus_area, [])
        similar_matches = []
        for similar_area in similar_areas:
            similar_normalized = similar_area.lower().replace('/', '_').replace(' ', '_')
            similar_col = f'focus_{similar_normalized}'
            if similar_col in poses_df.columns:
                similar_match = poses_df[poses_df[similar_col] == 1]
                if len(similar_match) > 0:
                    similar_matches.append(similar_match)
        if similar_matches:
            similar_matches_df = pd.concat(similar_matches).drop_duplicates()
            beginner_count = len(similar_matches_df[similar_matches_df['difficulty_score'] == 1])
            intermediate_count = len(similar_matches_df[similar_matches_df['difficulty_score'] == 2])
            advanced_count = len(similar_matches_df[similar_matches_df['difficulty_score'] == 3])
            print(f"   Focus area filter: {len(similar_matches_df)} similar matches found "
                  f"({beginner_count} Beginner, {intermediate_count} Intermediate, {advanced_count} Advanced)")
            return similar_matches_df
        print(f"   ⚠️ No matches found for focus area '{focus_area}' or similar areas. Returning all safe poses.")
        return poses_df

    def _filter_by_difficulty(self, poses_df, user_profile):
        """Filter poses by difficulty level appropriateness"""
        user_difficulty = user_profile.get('difficulty_score', 1)

        if user_difficulty == 1:  # Beginner
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 1]
        elif user_difficulty == 2:  # Intermediate
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 2]
        else:  # Advanced
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 3]

        print(f"   Difficulty filter: {len(appropriate_poses)} appropriate poses")
        return appropriate_poses

    def _calculate_recommendation_scores(self, poses_df, user_profile):
        """Calculate recommendation scores with emphasis on difficulty match"""
        scores = []
        for idx, pose in poses_df.iterrows():
            score = 0.0
            # Safety score (30% weight)
            score += pose['safety_score'] * 0.3
            # Focus area alignment (25% weight)
            focus_area = user_profile.get('focus_area', 'Flexibility/Stretching')
            focus_normalized = focus_area.lower().replace('/', '_').replace(' ', '_')
            focus_col = f'focus_{focus_normalized}'
            if focus_col in poses_df.columns and pose[focus_col] == 1:
                score += 0.25
            # Difficulty appropriateness (45% weight to further prioritize exact match)
            user_difficulty = user_profile.get('difficulty_score', 1)
            if pose['difficulty_score'] == user_difficulty:
                difficulty_score = 1.0
            elif abs(pose['difficulty_score'] - user_difficulty) == 1:
                difficulty_score = 0.2
            else:
                difficulty_score = 0.05
            score += difficulty_score * 0.45
            # Physical capability alignment (5% weight)
            fitness_composite = user_profile.get('fitness_composite', 50)
            if fitness_composite >= 60:
                score += 0.05
            elif fitness_composite >= 40:
                score += 0.025
            # Effectiveness bonus (10% weight)
            score += pose['effectiveness_score'] * 0.1
            scores.append(score)
        return scores

    def _get_neural_network_scores(self, poses_df, user_profile):
        """Get scores from neural network model"""
        if not hasattr(self, 'model'):
            print("   Neural network model not available, using rule-based scoring")
            return self._calculate_recommendation_scores(poses_df, user_profile)

        # Prepare user features
        user_feature_cols = ['age', 'bmi', 'flexibility_score', 'strength_score',
                             'balance_score', 'cardio_score', 'fitness_composite', 'difficulty_score']

        user_features = []
        for col in user_feature_cols:
            user_features.append(user_profile.get(col, 0))

        user_features = np.array(user_features).reshape(1, -1)
        user_features = self.scaler.transform(user_features)
        user_features = np.nan_to_num(user_features)

        # Prepare asana features
        asana_feature_cols = ['difficulty_score', 'complexity_score', 'safety_score',
                              'accessibility_score', 'effectiveness_score'] + \
                             self.focus_feature_columns + self.body_feature_columns

        asana_feature_cols = [col for col in asana_feature_cols if col in poses_df.columns]
        asana_features = poses_df[asana_feature_cols].values
        asana_features = np.nan_to_num(asana_features)

        # Get neural network scores
        self.model.eval()
        scores = []

        with torch.no_grad():
            for i in range(len(poses_df)):
                user_tensor = torch.FloatTensor(user_features).to(self.device)
                asana_tensor = torch.FloatTensor(asana_features[i:i + 1]).to(self.device)

                score = self.model(user_tensor, asana_tensor).cpu().numpy()[0][0]
                scores.append(score)

        return scores

    def recommend_poses(self, user_input, top_k=5):
        """
        Generate top-k yoga pose recommendations for a user
        """
        print(f"\n🔍 Generating {top_k} personalized recommendations...")

        # Validate user input
        required_fields = ['age', 'weight', 'height', 'focus_area', 'physical_level']
        missing_fields = [field for field in required_fields if field not in user_input]

        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return []

        # Process user input
        user_profile = self._process_user_input(user_input)

        # Ensure asanas_processed is available
        if not hasattr(self, 'asanas_processed'):
            print("⚠️ Asanas data not processed. Please load model or preprocess data.")
            return []

        # 1. Safety filtering (highest priority)
        safe_poses = self._filter_safe_poses(user_profile)

        if len(safe_poses) == 0:
            print("❌ No safe poses found for this user profile")
            return []

        # 2. Focus area filtering
        focus_filtered = self._filter_by_focus_area(safe_poses, user_profile)

        # 3. Difficulty level filtering
        difficulty_filtered = self._filter_by_difficulty(focus_filtered, user_profile)

        if len(difficulty_filtered) == 0:
            print("❌ No poses found matching all criteria")
            return []

        # 4. Calculate recommendation scores
        if hasattr(self, 'model'):
            scores = self._get_neural_network_scores(difficulty_filtered, user_profile)
        else:
            scores = self._calculate_recommendation_scores(difficulty_filtered, user_profile)

        # 5. Add scores to dataframe and sort
        difficulty_filtered = difficulty_filtered.copy()
        difficulty_filtered['recommendation_score'] = scores

        # Sort by score and get top recommendations
        top_recommendations = difficulty_filtered.nlargest(top_k, 'recommendation_score')

        # Format recommendations
        recommendations = []
        for idx, pose in top_recommendations.iterrows():
            recommendation = {
                'pose_name': pose['asana_name'],
                'difficulty_level': pose['difficulty_level'],
                'score': pose['recommendation_score'],
                'safety_score': pose['safety_score'],
                'focus_areas': self._get_focus_areas(pose),
                'body_parts': self._get_body_parts(pose),
                'precautions': self._get_precautions(pose),
                'benefits': self._get_benefits(pose),
                'recommendation_reason': self._get_recommendation_reason(pose, user_profile)
            }
            recommendations.append(recommendation)

        print(f"✅ Generated {len(recommendations)} personalized recommendations")
        return recommendations

    def _get_focus_areas(self, pose):
        """Extract focus areas for a pose"""
        focus_areas = []
        for col in self.focus_feature_columns:
            if pose[col] == 1:
                area = col.replace('focus_', '').replace('_', ' ').title()
                focus_areas.append(area)
        return focus_areas

    def _get_body_parts(self, pose):
        """Extract body parts for a pose"""
        body_parts = []
        for col in self.body_feature_columns:
            if pose[col] == 1:
                part = col.replace('body_', '').replace('_', ' ').title()
                body_parts.append(part)
        return body_parts

    def _get_precautions(self, pose):
        """Extract precautions for a pose"""
        precautions = []
        for col in self.precaution_columns:
            if pose[col] == 1:
                precaution = col.replace('precaution_', '').replace('_', ' ').title()
                precautions.append(precaution)
        return precautions

    def _get_benefits(self, pose):
        """Extract benefits for a pose"""
        benefits = []
        for col in self.pain_columns:
            if pose[col] == 1:
                benefit = col.replace('pain_', '').replace('_', ' ').title()
                benefits.append(f"Helps with {benefit}")
        return benefits

    def _get_recommendation_reason(self, pose, user_profile):
        """Generate explanation for why this pose was recommended"""
        reasons = []

        # Safety
        if pose['safety_score'] >= 0.8:
            reasons.append("Safe for your health profile")

        # Focus area match
        focus_area = user_profile.get('focus_area', 'Flexibility/Stretching')
        focus_normalized = focus_area.lower().replace('/', '_').replace(' ', '_')
        focus_col = f'focus_{focus_normalized}'

        if focus_col in pose.index and pose[focus_col] == 1:
            reasons.append(f"Matches your focus area: {focus_area}")

        # Difficulty appropriateness
        user_difficulty = user_profile.get('difficulty_score', 1)
        if abs(pose['difficulty_score'] - user_difficulty) <= 0.5:
            reasons.append("Appropriate for your skill level")

        # High effectiveness
        if pose['effectiveness_score'] >= 0.7:
            reasons.append("Highly effective pose")

        return "; ".join(reasons)

    def create_training_data(self):
        """Create training data for the neural network"""
        print("\n🔧 Creating training data...")

        # Create user-asana pairs
        user_features = []
        asana_features = []
        labels = []

        # User feature columns
        user_feature_cols = ['age', 'bmi', 'flexibility_score', 'strength_score',
                             'balance_score', 'cardio_score', 'fitness_composite', 'difficulty_score']

        # Add difficulty score for users
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.users_df['difficulty_score'] = self.users_df['yoga_difficulty_level'].map(difficulty_mapping)

        # Asana feature columns
        asana_feature_cols = ['difficulty_score', 'complexity_score', 'safety_score',
                              'accessibility_score', 'effectiveness_score'] + \
                             self.focus_feature_columns + self.body_feature_columns

        # Ensure all columns exist
        user_feature_cols = [col for col in user_feature_cols if col in self.users_df.columns]
        asana_feature_cols = [col for col in asana_feature_cols if col in self.asanas_processed.columns]

        # Create positive and negative samples
        n_samples = min(10000, len(self.users_df) * 20)  # Limit for memory

        for i in range(n_samples):
            user_idx = np.random.randint(0, len(self.users_df))
            asana_idx = np.random.randint(0, len(self.asanas_processed))

            user_row = self.users_df.iloc[user_idx]
            asana_row = self.asanas_processed.iloc[asana_idx]

            # Create feature vectors
            user_feat = user_row[user_feature_cols].values.astype(np.float32)
            asana_feat = asana_row[asana_feature_cols].values.astype(np.float32)

            # Create label based on compatibility
            label = self._calculate_compatibility(user_row, asana_row)

            user_features.append(user_feat)
            asana_features.append(asana_feat)
            labels.append(label)

        # Convert to numpy arrays
        user_features = np.array(user_features, dtype=np.float32)
        asana_features = np.array(asana_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)

        # Scale features
        user_features = self.scaler.fit_transform(user_features)

        # Handle NaN values
        user_features = np.nan_to_num(user_features)
        asana_features = np.nan_to_num(asana_features)

        # Store feature dimensions
        self.user_feature_dim = user_features.shape[1]
        self.asana_feature_dim = asana_features.shape[1]

        print(f"   ✅ Training data created: {len(labels)} samples")
        print(f"   ✅ User features: {self.user_feature_dim} dimensions")
        print(f"   ✅ Asana features: {self.asana_feature_dim} dimensions")

        return user_features, asana_features, labels

    def _calculate_compatibility(self, user_row, asana_row):
        score = 0.0
        if self._is_safe_for_user(user_row, asana_row):
            score += 0.35
        else:
            return 0.0
        user_difficulty = user_row.get('difficulty_score', 1)
        asana_difficulty = asana_row.get('difficulty_score', 1)
        if user_difficulty == asana_difficulty:
            score += 0.30
        elif abs(user_difficulty - asana_difficulty) == 1:
            score += 0.15
        else:
            score += 0.06
        user_age = user_row.get('age', 35)
        if 8 <= user_age <= 80:
            score += 0.10
        user_fitness = user_row.get('fitness_composite', 50)
        if user_fitness >= 60:
            score += 0.05
        elif user_fitness >= 40:
            score += 0.025
        score += asana_row.get('effectiveness_score', 0.5) * 0.10
        return min(score, 1.0)

    def _is_safe_for_user(self, user_row, asana_row):
        """Check if asana is safe for user based on health conditions"""
        if user_row.get('bp_risk') == 'high_bp':
            if asana_row.get('precaution_high_blood_pressure', 0) == 1:
                return False

        if user_row.get('bp_risk') == 'low_bp':
            if asana_row.get('precaution_low_blood_pressure', 0) == 1:
                return False

        if user_row.get('age', 35) > 65:
            if asana_row.get('complexity_score', 1) > 2.5:
                return False

        if user_row.get('bmi', 25) > 35:
            if asana_row.get('precaution_back_injuries', 0) == 1:
                return False

        return True

    def train_neural_network(self):
        """Train the neural network recommendation model"""
        print("\n🚀 Training Neural Network Model...")

        user_features, asana_features, labels = self.create_training_data()

        train_size = int(0.8 * len(labels))
        train_user_feat = user_features[:train_size]
        train_asana_feat = asana_features[:train_size]
        train_labels = labels[:train_size]

        val_user_feat = user_features[train_size:]
        val_asana_feat = asana_features[train_size:]
        val_labels = labels[train_size:]

        train_dataset = YogaRecommendationDataset(train_user_feat, train_asana_feat, train_labels)
        val_dataset = YogaRecommendationDataset(val_user_feat, val_asana_feat, val_labels)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

        self.model = YogaRecommendationNN(
            user_feature_dim=self.user_feature_dim,
            asana_feature_dim=self.asana_feature_dim,
            hidden_dim=128,
            dropout=0.3
        ).to(self.device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        train_losses = []
        val_losses = []

        for epoch in range(20):
            self.model.train()
            train_loss = 0.0

            for user_feat, asana_feat, labels_batch in train_loader:
                user_feat = user_feat.to(self.device)
                asana_feat = asana_feat.to(self.device)
                labels_batch = labels_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(user_feat, asana_feat).squeeze()
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for user_feat, asana_feat, labels_batch in val_loader:
                    user_feat = user_feat.to(self.device)
                    asana_feat = asana_feat.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    outputs = self.model(user_feat, asana_feat).squeeze()
                    loss = criterion(outputs, labels_batch)
                    val_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            if epoch % 10 == 0:
                print(f"   Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        print("   ✅ Neural network training completed")

    def evaluate_model(self):
        """Evaluate the recommendation model using various metrics"""
        print("\n📊 Evaluating Recommendation Model...")

        if not hasattr(self, 'model'):
            print("❌ Neural network model not available for evaluation")
            return

        test_user_features, test_asana_features, test_labels = self.create_training_data()

        n_test = min(1000, len(test_labels))
        test_user_features = test_user_features[:n_test]
        test_asana_features = test_asana_features[:n_test]
        test_labels = test_labels[:n_test]

        self.model.eval()
        predictions = []

        with torch.no_grad():
            for i in range(n_test):
                user_tensor = torch.FloatTensor(test_user_features[i:i + 1]).to(self.device)
                asana_tensor = torch.FloatTensor(test_asana_features[i:i + 1]).to(self.device)

                pred = self.model(user_tensor, asana_tensor).cpu().numpy()[0][0]
                predictions.append(pred)

        predictions = np.array(predictions)

        binary_predictions = (predictions > 0.5).astype(int)
        binary_labels = (test_labels > 0.5).astype(int)

        precision = precision_score(binary_labels, binary_predictions, average='weighted')
        recall = recall_score(binary_labels, binary_predictions, average='weighted')
        f1 = f1_score(binary_labels, binary_predictions, average='weighted')

        mse = np.mean((predictions - test_labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - test_labels))

        ndcg = ndcg_score(test_labels.reshape(1, -1), predictions.reshape(1, -1), k=5)

        print(f"📈 Model Evaluation Results:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   NDCG@5: {ndcg:.4f}")

        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'rmse': rmse,
            'mae': mae,
            'ndcg': ndcg
        }

    def save_model(self, filepath='yoga_recommender_model_final.pkl'):
        """Save the trained model, preprocessors, and processed asana data"""
        print(f"\n💾 Saving model to {filepath}...")

        model_data = {
            'model_state_dict': self.model.state_dict() if hasattr(self, 'model') else None,
            'scaler': self.scaler,
            'mlb_focus': self.mlb_focus,
            'mlb_precautions': self.mlb_precautions,
            'mlb_pain_points': self.mlb_pain_points,
            'mlb_body_parts': self.mlb_body_parts,
            'user_feature_dim': getattr(self, 'user_feature_dim', None),
            'asana_feature_dim': getattr(self, 'asana_feature_dim', None),
            'focus_feature_columns': getattr(self, 'focus_feature_columns', []),
            'body_feature_columns': getattr(self, 'body_feature_columns', []),
            'precaution_columns': getattr(self, 'precaution_columns', []),
            'pain_columns': getattr(self, 'pain_columns', []),
            'asanas_processed': getattr(self, 'asanas_processed', None)  # Save processed asana data
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Model and asana data saved to {filepath}")

    def load_model(self, filepath='yoga_recommender_model_final.pkl'):
        """Load a pre-trained model and processed asana data"""
        print(f"\n📂 Loading model from {filepath}...")

        try:
            # Set pickle to load tensors on CPU when CUDA is not available
            if self.device.type == 'cpu':
                import torch
                original_load = torch.load
                torch.load = lambda f, **kwargs: original_load(f, map_location='cpu', **kwargs)
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
                
            # Restore original torch.load if we modified it
            if self.device.type == 'cpu':
                torch.load = original_load

            self.scaler = model_data['scaler']
            self.mlb_focus = model_data['mlb_focus']
            self.mlb_precautions = model_data['mlb_precautions']
            self.mlb_pain_points = model_data['mlb_pain_points']
            self.mlb_body_parts = model_data['mlb_body_parts']

            self.focus_feature_columns = model_data['focus_feature_columns']
            self.body_feature_columns = model_data['body_feature_columns']
            self.precaution_columns = model_data['precaution_columns']
            self.pain_columns = model_data['pain_columns']

            # Load asanas_processed if available
            if 'asanas_processed' in model_data and model_data['asanas_processed'] is not None:
                self.asanas_processed = model_data['asanas_processed']
                print(f"✅ Loaded processed asana data: {self.asanas_processed.shape}")

            if model_data['model_state_dict'] is not None:
                self.user_feature_dim = model_data['user_feature_dim']
                self.asana_feature_dim = model_data['asana_feature_dim']

                self.model = YogaRecommendationNN(
                    user_feature_dim=self.user_feature_dim,
                    asana_feature_dim=self.asana_feature_dim
                ).to(self.device)

                # Load model state dict with CPU mapping to handle CUDA->CPU loading
                state_dict = model_data['model_state_dict']
                if self.device.type == 'cpu':
                    # If we're on CPU, map all tensors to CPU
                    state_dict = {key: value.cpu() if hasattr(value, 'cpu') else value for key, value in state_dict.items()}
                self.model.load_state_dict(state_dict)
                self.model.eval()

            print("✅ Model loaded successfully")

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            if 'asanas_processed' not in model_data:
                print(
                    "⚠️ Processed asana data not found in pickle file. Please preprocess data or use a compatible model file.")

    def get_user_statistics(self):
        """Get statistics about the user dataset"""
        print("\n📊 User Dataset Statistics:")
        print(f"Total users: {len(self.users_df)}")
        print(f"Age range: {self.users_df['age'].min():.1f} - {self.users_df['age'].max():.1f}")
        print(f"BMI range: {self.users_df['bmi'].min():.1f} - {self.users_df['bmi'].max():.1f}")
        print(f"Fitness levels: {self.users_df['class'].value_counts().to_dict()}")
        print(f"Risk levels: {self.users_df['practice_risk_level'].value_counts().to_dict()}")

    def get_asana_statistics(self):
        """Get statistics about the asana dataset"""
        print("\n📊 Asana Dataset Statistics:")
        print(f"Total asanas: {len(self.asanas_processed)}")
        print(f"Difficulty levels: {self.asanas_processed['difficulty_level'].value_counts().to_dict()}")
        print(f"Average safety score: {self.asanas_processed['safety_score'].mean():.3f}")
        print(f"Average accessibility score: {self.asanas_processed['accessibility_score'].mean():.3f}")


class YogaRecommendationNN(nn.Module):
    """Neural Network for Yoga Pose Recommendation"""

    def __init__(self, user_feature_dim, asana_feature_dim, hidden_dim=128, dropout=0.3):
        super(YogaRecommendationNN, self).__init__()

        self.user_encoder = nn.Sequential(
            nn.Linear(user_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.asana_encoder = nn.Sequential(
            nn.Linear(asana_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, user_features, asana_features):
        user_embedding = self.user_encoder(user_features)
        asana_embedding = self.asana_encoder(asana_features)

        combined = torch.cat([user_embedding, asana_embedding], dim=-1)
        score = self.interaction_layer(combined)
        return score


class YogaRecommendationDataset(Dataset):
    """PyTorch Dataset for Yoga Recommendations"""

    def __init__(self, user_features, asana_features, labels):
        self.user_features = torch.FloatTensor(user_features)
        self.asana_features = torch.FloatTensor(asana_features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_features[idx], self.asana_features[idx], self.labels[idx]


def main():
    """Main function to run the Yoga Recommender System"""
    # Initialize the recommender system
    recommender = YogaRecommenderSystem()

    # Load and preprocess data
    recommender.load_and_preprocess_data()

    # Display dataset statistics
    recommender.get_user_statistics()
    recommender.get_asana_statistics()

    # Train the neural network model
    recommender.train_neural_network()

    # Evaluate the model
    recommender.evaluate_model()

    # Sample user input for recommendations
    user_input = {
        'age': 35,
        'weight': 70,
        'height': 170,
        'focus_area': 'Balance/Stability',
        'physical_level': 'Advanced',
        'health_conditions': [],
        'injuries': [],
        'bp_systolic': 120,
        'bp_diastolic': 80
    }

    # Generate and print recommendations
    recommendations = recommender.recommend_poses(user_input, top_k=5)
    print("\n📋 Sample User Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"\nRecommendation {i}:")
        print(f"Pose: {rec['pose_name']}")
        print(f"Difficulty: {rec['difficulty_level']}")
        print(f"Score: {rec['score']:.4f}")
        print(f"Safety Score: {rec['safety_score']:.4f}")
        print(f"Focus Areas: {', '.join(rec['focus_areas'])}")
        print(f"Body Parts: {', '.join(rec['body_parts'])}")
        print(f"Precautions: {', '.join(rec['precautions']) if rec['precautions'] else 'None'}")
        print(f"Benefits: {', '.join(rec['benefits']) if rec['benefits'] else 'General wellness'}")
        print(f"Reason: {rec['recommendation_reason']}")

    # Save the trained model
    recommender.save_model()


if __name__ == "__main__":
    main()