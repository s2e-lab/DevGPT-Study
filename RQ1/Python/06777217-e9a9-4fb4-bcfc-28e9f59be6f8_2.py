# models.py

from django.db import models
from django_ltree.fields import LtreeField

class TreeNode(models.Model):
    name = models.CharField(max_length=100)
    path = LtreeField()
    parent = models.ForeignKey('self', null=True, blank=True, on_delete=models.CASCADE)

    def __str__(self):
        return self.name
