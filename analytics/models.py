from django.db import models

# Create your models here.
from django.db import models

class ChatbotAnalytics(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    response_time = models.FloatField()  # Time taken to respond in seconds

    def __str__(self):
        return f"User: {self.user_message[:30]}... | Response Time: {self.response_time}s"
