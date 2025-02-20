from django.shortcuts import render

# Create your views here.
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import ChatbotAnalytics
from .serializers import ChatbotAnalyticsSerializer

class ChatbotAnalyticsAPI(APIView):
    def get(self, request):
        analytics = ChatbotAnalytics.objects.all().order_by("-timestamp")[:10]
        serializer = ChatbotAnalyticsSerializer(analytics, many=True)
        return Response(serializer.data)
