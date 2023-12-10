from django.urls import path
from .views import my_view, MyView

urlpatterns = [
    path('my_view/', my_view, name='my_view'),
    path('my_class_view/', MyView.as_view(), name='my_class_view'),
]
