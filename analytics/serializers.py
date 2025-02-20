from rest_framework import serializers
from .models import ChatbotAnalytics

class ChatbotAnalyticsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatbotAnalytics
        fields = '__all__'
