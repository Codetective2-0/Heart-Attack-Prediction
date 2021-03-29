from django.db import models

# Create your models here.
class Contact(models.Model):
     name= models.CharField(max_length=255)
     email = models.CharField(max_length=100)
     subject = models.CharField(max_length=255)
     content= models.TextField()

     timeStamp=models.DateTimeField(auto_now_add=True, blank=True)