# Data processing utilities
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import warnings

warnings.filterwarnings('ignore')

class YogaDataPreprocessor:
    """Handles data loading and preprocessing for the Yoga Recommender System"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.mlb_focus = MultiLabelBinarizer()
        self.mlb_precautions = MultiLabelBinarizer()
        self.mlb_pain_points = MultiLabelBinarizer()
        self.mlb_body_parts = MultiLabelBinarizer()

        # Safety mappings
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

        # Fitness to yoga difficulty mapping
        self.fitness_to_yoga_mapping = {
            'A': 'Advanced',
            'B': 'Intermediate',
            'C': 'Beginner',
            'D': 'Beginner'
        }

        print("🧘 Data Preprocessor Initialized")

    def load_and_preprocess_data(self):
        """Load and preprocess both datasets"""
        print("\n📊 Loading and preprocessing datasets...")
        try:
            self.users_df = pd.read_csv('/kaggle/input/users-data/bodyPerformance.csv')
            self.asanas_df = pd.read_csv('/kaggle/input/final-asana-dataset/new_final_dataset (1).csv')
            print(f"✅ Users dataset loaded: {self.users_df.shape}")
            print(f"✅ Asanas dataset loaded: {self.asanas_df.shape}")
        except FileNotFoundError:
            print("⚠️ Kaggle datasets not found, creating synthetic data...")
            self._create_synthetic_data()
        self._preprocess_users_data()
        self._preprocess_asanas_data()
        print("✅ Data preprocessing completed")
        return self.users_df, self.asanas_processed

    def _create_synthetic_data(self):
        """Create synthetic data for testing purposes"""
        np.random.seed(42)
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
            'Pain_Points': np.random.choice(['Lower back pain', 'Stress/anxiety', 'Balance issues', 'Hip tightness'], n_asanas),
            'duration_secs': np.random.randint(30, 180, n_asanas),
            'target_age_group': ['8-80'] * n_asanas
        })

    def _preprocess_users_data(self):
        """Preprocess users dataset with comprehensive feature engineering"""
        print("   Processing users data...")
        self.users_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for col in self.users_df.columns]
        numeric_cols = self.users_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            self.users_df[col].fillna(self.users_df[col].median(), inplace=True)
        self.users_df['bmi'] = self.users_df['weight_kg'] / (self.users_df['height_cm'] / 100) ** 2
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
        self.users_df['flexibility_score'] = np.clip((self.users_df.get('sit_and_bend_forward_cm', 15) + 10) * 2.5, 0, 100)
        self.users_df['strength_score'] = np.clip(self.users_df.get('gripforce', 35) * 2, 0, 100)
        self.users_df['balance_score'] = np.clip(self.users_df.get('broad_jump_cm', 180) / 3, 0, 100)
        self.users_df['cardio_score'] = np.clip(100 - self.users_df.get('body_fat_%', 20), 0, 100)
        self.users_df['yoga_difficulty_level'] = self.users_df['class'].map(self.fitness_to_yoga_mapping)
        self.users_df['fitness_composite'] = (
                self.users_df['flexibility_score'] * 0.25 +
                self.users_df['strength_score'] * 0.25 +
                self.users_df['balance_score'] * 0.25 +
                self.users_df['cardio_score'] * 0.25
        )
        self.users_df['age_group'] = pd.cut(
            self.users_df['age'],
            bins=[0, 30, 45, 60, 100],
            labels=['Young', 'Adult', 'Middle_Age', 'Senior']
        )
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
        self.users_df.loc[medium_risk & (self.users_df['practice_risk_level'] == 'low'), 'practice_risk_level'] = 'medium'
        print(f"   ✅ Users processed: {self.users_df.shape}")

    def _preprocess_asanas_data(self):
        """Preprocess asanas dataset with advanced feature engineering"""
        print("   Processing asanas data...")
        self.asanas_df.columns = [col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_') for col in self.asanas_df.columns]
        self.asanas_df['asana_name'].fillna('Unknown_Pose', inplace=True)
        self.asanas_df['difficulty_level'].fillna('Beginner', inplace=True)
        def parse_multi_value(field_value):
            if pd.isna(field_value):
                return []
            items = str(field_value).split(',')
            return [item.strip().replace('[', '').replace(']', '').replace("'", "") for item in items if item.strip()]
        self.asanas_df['focus_area_list'] = self.asanas_df.get('focus_area', 'Flexibility/Stretching').apply(parse_multi_value)
        self.asanas_df['body_parts_list'] = self.asanas_df.get('body_parts', 'Full Body').apply(parse_multi_value)
        self.asanas_df['precautions_list'] = self.asanas_df.get('precautions', 'Generally safe for all').apply(parse_multi_value)
        self.asanas_df['pain_points_list'] = self.asanas_df.get('pain_points', 'General wellness').apply(parse_multi_value)
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.asanas_df['difficulty_score'] = self.asanas_df['difficulty_level'].map(difficulty_mapping)
        self.asanas_df['complexity_score'] = self.asanas_df['difficulty_score'].copy()
        focus_matrix = self.mlb_focus.fit_transform(self.asanas_df['focus_area_list'])
        focus_columns = [f'focus_{area.lower().replace("/", "_").replace(" ", "_")}' for area in self.mlb_focus.classes_]
        focus_df = pd.DataFrame(focus_matrix, columns=focus_columns, index=self.asanas_df.index)
        body_matrix = self.mlb_body_parts.fit_transform(self.asanas_df['body_parts_list'])
        body_columns = [f'body_{part.lower().replace("/", "_").replace(" ", "_")}' for part in self.mlb_body_parts.classes_]
        body_df = pd.DataFrame(body_matrix, columns=body_columns, index=self.asanas_df.index)
        precautions_matrix = self.mlb_precautions.fit_transform(self.asanas_df['precautions_list'])
        precautions_columns = [f'precaution_{prec.lower().replace("/", "_").replace(" ", "_")}' for prec in self.mlb_precautions.classes_]
        precautions_df = pd.DataFrame(precautions_matrix, columns=precautions_columns, index=self.asanas_df.index)
        pain_matrix = self.mlb_pain_points.fit_transform(self.asanas_df['pain_points_list'])
        pain_columns = [f'pain_{point.lower().replace("/", "_").replace(" ", "_")}' for point in self.mlb_pain_points.classes_]
        pain_df = pd.DataFrame(pain_matrix, columns=pain_columns, index=self.asanas_df.index)
        self.asanas_processed = pd.concat([
            self.asanas_df[['asana_name', 'difficulty_level', 'difficulty_score', 'complexity_score']],
            focus_df, body_df, precautions_df, pain_df
        ], axis=1)
        self.asanas_processed['safety_score'] = 1.0
        risky_precautions = ['high_blood_pressure', 'heart_conditions', 'back_injuries',
                             'knee_problems_injuries', 'neck_injuries', 'pregnancy']
        for precaution in risky_precautions:
            precaution_col = f'precaution_{precaution}'
            if precaution_col in self.asanas_processed.columns:
                self.asanas_processed.loc[self.asanas_processed[precaution_col] == 1, 'safety_score'] -= 0.2
        self.asanas_processed['accessibility_score'] = 1.0
        focus_columns = [col for col in self.asanas_processed.columns if col.startswith('focus_')]
        self.asanas_processed['effectiveness_score'] = self.asanas_processed[focus_columns].sum(axis=1) / max(len(focus_columns), 1)
        self.focus_feature_columns = focus_columns
        self.body_feature_columns = [col for col in self.asanas_processed.columns if col.startswith('body_')]
        self.precaution_columns = [col for col in self.asanas_processed.columns if col.startswith('precaution_')]
        self.pain_columns = [col for col in self.asanas_processed.columns if col.startswith('pain_')]
        print(f"   ✅ Asanas processed: {self.asanas_processed.shape}")
        print(f"   ✅ Focus features: {len(self.focus_feature_columns)}")
        print(f"   ✅ Safety features: {len(self.precaution_columns)}")

    def get_feature_columns(self):
        """Return feature column names for downstream tasks"""
        return {
            'focus_feature_columns': self.focus_feature_columns,
            'body_feature_columns': self.body_feature_columns,
            'precaution_columns': self.precaution_columns,
            'pain_columns': self.pain_columns
        }