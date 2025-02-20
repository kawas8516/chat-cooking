from django.db import models

# Create your models here.
from django.contrib.auth.models import AbstractUser, Group, Permission


class CustomUser(AbstractUser):
    dietary_preferences = models.JSONField(default=dict)  # e.g., {"vegan": True, "gluten_free": False}
    saved_recipes = models.ManyToManyField("recipes.Recipe", blank=True)
    preferred_cuisine = models.CharField(max_length=255, blank=True, null=True)

    groups = models.ManyToManyField(
        Group,
        related_name="custom_user_groups",  # Change related_name to avoid conflict
        blank=True,
    )
    user_permissions = models.ManyToManyField(
        Permission,
        related_name="custom_user_permissions",  # Change related_name to avoid conflict
        blank=True,
    )
