from django.db import models

class UserEmail(models.Model):
    subject = models.CharField(max_length=255)
    message = models.TextField()
