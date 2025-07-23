import pandas as pd
import numpy as np
import torch
from model import YogaRecommendationNN

class YogaRecommender:
    """Core recommendation logic for the Yoga Recommender System"""

    def __init__(self, users_df, asanas_processed, scaler, mlb_focus, mlb_precautions, mlb_pain_points, mlb_body_parts, feature_columns):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.users_df = users_df
        self.asanas_processed = asanas_processed
        self.scaler = scaler
        self.mlb_focus = mlb_focus
        self.mlb_precautions = mlb_precautions
        self.mlb_pain_points = mlb_pain_points
        self.mlb_body_parts = mlb_body_parts
        self.focus_feature_columns = feature_columns['focus_feature_columns']
        self.body_feature_columns = feature_columns['body_feature_columns']
        self.precaution_columns = feature_columns['precaution_columns']
        self.pain_columns = feature_columns['pain_columns']
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
        print("🧘 Yoga Recommender Initialized")

    def _process_user_input(self, user_input):
        """Process and validate user input for recommendations"""
        user_profile = {}
        user_profile['age'] = user_input.get('age', 35)
        user_profile['weight'] = user_input.get('weight', 70)
        user_profile['height'] = user_input.get('height', 170)
        user_profile['gender'] = user_input.get('gender', 'M')
        user_profile['bmi'] = user_profile['weight'] / (user_profile['height'] / 100) ** 2
        user_profile['physical_level'] = user_input.get('physical_level', 'Beginner')
        user_profile['yoga_experience'] = user_input.get('yoga_experience', 'Beginner')
        user_profile['focus_area'] = user_input.get('focus_area', 'Flexibility/Stretching')
        user_profile['health_conditions'] = user_input.get('health_conditions', [])
        user_profile['injuries'] = user_input.get('injuries', [])
        user_profile['bp_systolic'] = user_input.get('bp_systolic', 120)
        user_profile['bp_diastolic'] = user_input.get('bp_diastolic', 80)
        if user_profile['bp_systolic'] > 140 or user_profile['bp_diastolic'] > 90:
            user_profile['bp_risk'] = 'high_bp'
        elif user_profile['bp_systolic'] < 90 or user_profile['bp_diastolic'] < 60:
            user_profile['bp_risk'] = 'low_bp'
        else:
            user_profile['bp_risk'] = 'normal_bp'
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
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        user_profile['difficulty_score'] = difficulty_mapping.get(user_profile['physical_level'], 1)
        return user_profile

    def _filter_safe_poses(self, user_profile):
        """Filter poses based on safety conditions"""
        if self.asanas_processed is None or self.asanas_processed.empty:
            print("❌ No asana data available for filtering")
            return pd.DataFrame()
        safe_poses = self.asanas_processed.copy()
        all_conditions = user_profile.get('health_conditions', []) + user_profile.get('injuries', [])
        for condition in all_conditions:
            condition_normalized = condition.lower().replace(' ', '_').replace('/', '_')
            precaution_col = f'precaution_{condition_normalized}'
            if precaution_col in safe_poses.columns:
                safe_poses = safe_poses[safe_poses[precaution_col] == 0]
        if user_profile.get('bp_risk') == 'high_bp':
            if 'precaution_high_blood_pressure' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_high_blood_pressure'] == 0]
        if user_profile.get('bp_risk') == 'low_bp':
            if 'precaution_low_blood_pressure' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_low_blood_pressure'] == 0]
        if user_profile.get('age', 35) > 65:
            safe_poses = safe_poses[safe_poses['complexity_score'] <= 2.0]
        if user_profile.get('bmi', 25) > 35:
            if 'precaution_back_injuries' in safe_poses.columns:
                safe_poses = safe_poses[safe_poses['precaution_back_injuries'] == 0]
        print(f"   Safety filter: {len(safe_poses)} poses remaining from {len(self.asanas_processed)}")
        return safe_poses

    def _filter_by_focus_area(self, poses_df, user_profile):
        """Filter poses by user's focus area with robust matching"""
        if poses_df.empty:
            print("❌ No poses to filter by focus area")
            return poses_df
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
        if poses_df.empty:
            print("❌ No poses to filter by difficulty")
            return poses_df
        user_difficulty = user_profile.get('difficulty_score', 1)
        if user_difficulty == 1:
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 1]
        elif user_difficulty == 2:
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 2]
        else:
            appropriate_poses = poses_df[poses_df['difficulty_score'] <= 3]
        print(f"   Difficulty filter: {len(appropriate_poses)} appropriate poses")
        return appropriate_poses

    def _calculate_recommendation_scores(self, poses_df, user_profile):
        """Calculate recommendation scores with emphasis on difficulty match"""
        if poses_df.empty:
            print("❌ No poses to score")
            return []
        scores = []
        for idx, pose in poses_df.iterrows():
            score = 0.0
            score += pose['safety_score'] * 0.3
            focus_area = user_profile.get('focus_area', 'Flexibility/Stretching')
            focus_normalized = focus_area.lower().replace('/', '_').replace(' ', '_')
            focus_col = f'focus_{focus_normalized}'
            if focus_col in poses_df.columns and pose[focus_col] == 1:
                score += 0.25
            user_difficulty = user_profile.get('difficulty_score', 1)
            if pose['difficulty_score'] == user_difficulty:
                difficulty_score = 1.0
            elif abs(pose['difficulty_score'] - user_difficulty) == 1:
                difficulty_score = 0.2
            else:
                difficulty_score = 0.05
            score += difficulty_score * 0.45
            fitness_composite = user_profile.get('fitness_composite', 50)
            if fitness_composite >= 60:
                score += 0.05
            elif fitness_composite >= 40:
                score += 0.025
            score += pose['effectiveness_score'] * 0.1
            scores.append(score)
        return scores

    def _get_neural_network_scores(self, poses_df, user_profile):
        """Get scores from neural network model"""
        if not hasattr(self, 'model'):
            print("   Neural network model not available, using rule-based scoring")
            return self._calculate_recommendation_scores(poses_df, user_profile)
        if poses_df.empty:
            print("❌ No poses to score with neural network")
            return []
        user_feature_cols = ['age', 'bmi', 'flexibility_score', 'strength_score',
                             'balance_score', 'cardio_score', 'fitness_composite', 'difficulty_score']
        user_features = []
        for col in user_feature_cols:
            user_features.append(user_profile.get(col, 0))
        user_features = np.array(user_features).reshape(1, -1)
        try:
            user_features = self.scaler.transform(user_features)
        except Exception as e:
            print(f"❌ Error scaling user features: {e}")
            return self._calculate_recommendation_scores(poses_df, user_profile)
        user_features = np.nan_to_num(user_features)
        asana_feature_cols = ['difficulty_score', 'complexity_score', 'safety_score',
                              'accessibility_score', 'effectiveness_score'] + \
                             self.focus_feature_columns + self.body_feature_columns
        asana_feature_cols = [col for col in asana_feature_cols if col in poses_df.columns]
        asana_features = poses_df[asana_feature_cols].values
        asana_features = np.nan_to_num(asana_features)
        self.model.eval()
        scores = []
        try:
            with torch.no_grad():
                for i in range(len(poses_df)):
                    user_tensor = torch.FloatTensor(user_features).to(self.device)
                    asana_tensor = torch.FloatTensor(asana_features[i:i + 1]).to(self.device)
                    score = self.model(user_tensor, asana_tensor).cpu().numpy()[0][0]
                    scores.append(float(score))  # Ensure scalar for pandas compatibility
            return scores
        except Exception as e:
            print(f"❌ Error in neural network scoring: {e}")
            return self._calculate_recommendation_scores(poses_df, user_profile)

    def recommend_poses(self, user_input, top_k=5):
        """Generate top-k yoga pose recommendations for a user"""
        print(f"\n🔍 Generating {top_k} personalized recommendations...")
        required_fields = ['age', 'weight', 'height', 'focus_area', 'physical_level']
        missing_fields = [field for field in required_fields if field not in user_input]
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return []
        user_profile = self._process_user_input(user_input)
        safe_poses = self._filter_safe_poses(user_profile)
        if len(safe_poses) == 0:
            print("❌ No safe poses found for this user profile")
            return []
        focus_filtered = self._filter_by_focus_area(safe_poses, user_profile)
        difficulty_filtered = self._filter_by_difficulty(focus_filtered, user_profile)
        if len(difficulty_filtered) == 0:
            print("❌ No poses found matching all criteria")
            return []
        scores = self._get_neural_network_scores(difficulty_filtered, user_profile)
        if not scores:
            print("❌ No scores generated, returning empty recommendations")
            return []
        try:
            difficulty_filtered = difficulty_filtered.copy()
            difficulty_filtered['recommendation_score'] = pd.Series(scores, index=difficulty_filtered.index)
            top_recommendations = difficulty_filtered.nlargest(top_k, 'recommendation_score')
        except Exception as e:
            print(f"❌ Error assigning scores to dataframe: {e}")
            return []
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
            if col in pose.index and pose[col] == 1:
                area = col.replace('focus_', '').replace('_', ' ').title()
                focus_areas.append(area)
        return focus_areas

    def _get_body_parts(self, pose):
        """Extract body parts for a pose"""
        body_parts = []
        for col in self.body_feature_columns:
            if col in pose.index and pose[col] == 1:
                part = col.replace('body_', '').replace('_', ' ').title()
                body_parts.append(part)
        return body_parts

    def _get_precautions(self, pose):
        """Extract precautions for a pose"""
        precautions = []
        for col in self.precaution_columns:
            if col in pose.index and pose[col] == 1:
                precaution = col.replace('precaution_', '').replace('_', ' ').title()
                precautions.append(precaution)
        return precautions

    def _get_benefits(self, pose):
        """Extract benefits for a pose"""
        benefits = []
        for col in self.pain_columns:
            if col in pose.index and pose[col] == 1:
                benefit = col.replace('pain_', '').replace('_', ' ').title()
                benefits.append(f"Helps with {benefit}")
        return benefits

    def _get_recommendation_reason(self, pose, user_profile):
        """Generate explanation for why this pose was recommended"""
        reasons = []
        if pose['safety_score'] >= 0.8:
            reasons.append("Safe for your health profile")
        focus_area = user_profile.get('focus_area', 'Flexibility/Stretching')
        focus_normalized = focus_area.lower().replace('/', '_').replace(' ', '_')
        focus_col = f'focus_{focus_normalized}'
        if focus_col in pose.index and pose[focus_col] == 1:
            reasons.append(f"Matches your focus area: {focus_area}")
        user_difficulty = user_profile.get('difficulty_score', 1)
        if abs(pose['difficulty_score'] - user_difficulty) <= 0.5:
            reasons.append("Appropriate for your skill level")
        if pose['effectiveness_score'] >= 0.7:
            reasons.append("Highly effective pose")
        return "; ".join(reasons)

    def create_training_data(self):
        """Create training data for the neural network"""
        print("\n🔧 Creating training data...")
        user_features = []
        asana_features = []
        labels = []
        user_feature_cols = ['age', 'bmi', 'flexibility_score', 'strength_score',
                             'balance_score', 'cardio_score', 'fitness_composite', 'difficulty_score']
        difficulty_mapping = {'Beginner': 1, 'Intermediate': 2, 'Advanced': 3}
        self.users_df['difficulty_score'] = self.users_df['yoga_difficulty_level'].map(difficulty_mapping)
        asana_feature_cols = ['difficulty_score', 'complexity_score', 'safety_score',
                              'accessibility_score', 'effectiveness_score'] + \
                             self.focus_feature_columns + self.body_feature_columns
        user_feature_cols = [col for col in user_feature_cols if col in self.users_df.columns]
        asana_feature_cols = [col for col in asana_feature_cols if col in self.asanas_processed.columns]
        n_samples = min(10000, len(self.users_df) * 20)
        for i in range(n_samples):
            user_idx = np.random.randint(0, len(self.users_df))
            asana_idx = np.random.randint(0, len(self.asanas_processed))
            user_row = self.users_df.iloc[user_idx]
            asana_row = self.asanas_processed.iloc[asana_idx]
            user_feat = user_row[user_feature_cols].values.astype(np.float32)
            asana_feat = asana_row[asana_feature_cols].values.astype(np.float32)
            label = self._calculate_compatibility(user_row, asana_row)
            user_features.append(user_feat)
            asana_features.append(asana_feat)
            labels.append(label)
        user_features = np.array(user_features, dtype=np.float32)
        asana_features = np.array(asana_features, dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        user_features = self.scaler.fit_transform(user_features)
        user_features = np.nan_to_num(user_features)
        asana_features = np.nan_to_num(asana_features)
        self.user_feature_dim = user_features.shape[1]
        self.asana_feature_dim = asana_features.shape[1]
        print(f"   ✅ Training data created: {len(labels)} samples")
        print(f"   ✅ User features: {self.user_feature_dim} dimensions")
        print(f"   ✅ Asana features: {self.asana_feature_dim} dimensions")
        return user_features, asana_features, labels

    def _calculate_compatibility(self, user_row, asana_row):
        """Calculate compatibility score for user-asana pair"""
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

    def set_model(self, model):
        """Set the neural network model"""
        self.model = model

    def load_model(self, filepath='yoga_recommender_model_final.pth'):
        """Load a pre-trained model"""
        print(f"\n📂 Loading model from {filepath}...")
        try:
            model_data = torch.load(filepath, map_location=self.device)
            self.scaler = model_data['scaler']
            self.mlb_focus = model_data['mlb_focus']
            self.mlb_precautions = model_data['mlb_precautions']
            self.mlb_pain_points = model_data['mlb_pain_points']
            self.mlb_body_parts = model_data['mlb_body_parts']
            self.focus_feature_columns = model_data['focus_feature_columns']
            self.body_feature_columns = model_data['body_feature_columns']
            self.precaution_columns = model_data['precaution_columns']
            self.pain_columns = model_data['pain_columns']
            if model_data['model_state_dict'] is not None:
                self.user_feature_dim = model_data['user_feature_dim']
                self.asana_feature_dim = model_data['asana_feature_dim']
                self.model = YogaRecommendationNN(
                    user_feature_dim=self.user_feature_dim,
                    asana_feature_dim=self.asana_feature_dim
                ).to(self.device)
                self.model.load_state_dict(model_data['model_state_dict'])
                self.model.eval()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")