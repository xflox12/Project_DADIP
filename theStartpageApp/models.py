from django.db import models

# Create your models here.

class Test(models.Model):
    test_id = models.IntegerField(default=0)
    test_text = models.CharField(max_length=200)