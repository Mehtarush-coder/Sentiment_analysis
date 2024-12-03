from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('update-plot', views.update_plot_route, name='update_plot'),
    # path('progress/', views.progress, name='progress'),
]

