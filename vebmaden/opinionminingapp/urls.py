from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze-sentiment/', views.analyze_sentiment_view, name='sentiment_analysis'),
    path('topic/', views.topic_modeling_view, name='topic_modeling'), 
    path('scrape-wikipedia/', views.scrape_wikipedia_view, name='scrape_wikipedia'),
]