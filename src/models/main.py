import torch
import pickle
from ..data import YogaDataPreprocessor
from recommender import YogaRecommender
from train_evaluate import YogaModelTrainer
import numpy as np

def main():
    """Main function to run the Yoga Recommender System"""
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)

    # Initialize components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data preprocessing
    preprocessor = YogaDataPreprocessor()
    users_df, asanas_processed = preprocessor.load_and_preprocess_data()

    # Initialize recommender
    recommender = YogaRecommender(
        users_df=users_df,
        asanas_processed=asanas_processed,
        scaler=preprocessor.scaler,
        mlb_focus=preprocessor.mlb_focus,
        mlb_precautions=preprocessor.mlb_precautions,
        mlb_pain_points=preprocessor.mlb_pain_points,
        mlb_body_parts=preprocessor.mlb_body_parts,
        feature_columns=preprocessor.get_feature_columns()
    )

    # Create training data
    user_features, asana_features, labels = recommender.create_training_data()

    # Split data
    train_size = int(0.8 * len(labels))
    train_user_features = user_features[:train_size]
    train_asana_features = asana_features[:train_size]
    train_labels = labels[:train_size]
    val_user_features = user_features[train_size:]
    val_asana_features = asana_features[train_size:]
    val_labels = labels[train_size:]

    # Train and evaluate model
    trainer = YogaModelTrainer(
        user_feature_dim=recommender.user_feature_dim,
        asana_feature_dim=recommender.asana_feature_dim,
        device=device
    )
    trainer.train_neural_network(
        train_user_features, train_asana_features, train_labels,
        val_user_features, val_asana_features, val_labels
    )
    recommender.set_model(trainer.model)
    trainer.evaluate_model(user_features, asana_features, labels)

    # Display statistics
    preprocessor.get_user_statistics()
    preprocessor.get_asana_statistics()

    # Sample user input
    user_input = {
        'age': 30,
        'weight': 70,
        'height': 170,
        'focus_area': 'Balance/Stability',
        'physical_level': 'Advanced',
        'health_conditions': [],
        'injuries': [],
        'bp_systolic': 120,
        'bp_diastolic': 80
    }

    # Generate recommendations
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

    # Save model
    model_data = {
        'model_state_dict': trainer.model.state_dict(),
        'scaler': preprocessor.scaler,
        'mlb_focus': preprocessor.mlb_focus,
        'mlb_precautions': preprocessor.mlb_precautions,
        'mlb_pain_points': preprocessor.mlb_pain_points,
        'mlb_body_parts': preprocessor.mlb_body_parts,
        'user_feature_dim': recommender.user_feature_dim,
        'asana_feature_dim': recommender.asana_feature_dim,
        'focus_feature_columns': recommender.focus_feature_columns,
        'body_feature_columns': recommender.body_feature_columns,
        'precaution_columns': recommender.precaution_columns,
        'pain_columns': recommender.pain_columns
    }
    torch.save(model_data, 'yoga_recommender_model_final.pth')
    with open('yoga_recommender_model_final.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    print(f"✅ Model saved to yoga_recommender_model_final.pth and yoga_recommender_model_final.pkl")


if __name__ == "__main__":
    main()