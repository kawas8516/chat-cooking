from django.shortcuts import render

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from .utils import recommend_recipes

class RecipeRecommendationAPIView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        user_ingredients = request.data.get("ingredients", [])
        recommendations = recommend_recipes(user_ingredients)
        return Response({"recipes": [recipe.name for recipe in recommendations]})
