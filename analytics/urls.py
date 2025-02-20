from django.urls import path
from .views import ChatbotAnalyticsAPI

urlpatterns = [
    path('analytics/', ChatbotAnalyticsAPI.as_view(), name="chatbot-analytics-api"),
]
