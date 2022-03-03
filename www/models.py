from django.db import models

# Create your models here.
class Anti(models.Model):
	idx = models.CharField(max_length=10)

	def __str__(self):
		return self.name