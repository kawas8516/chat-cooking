from django.urls import path
from .views import RecipeRecommendationAPIView

urlpatterns = [
    path("recommend/", RecipeRecommendationAPIView.as_view(), name="recommend_recipes"),
]
