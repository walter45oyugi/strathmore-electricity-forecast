from django.urls import path

from .views import ForecastAPI, forecast_view, overview


urlpatterns = [
    path("", overview, name="overview"),
    path("forecast/", forecast_view, name="forecast"),
    path("api/forecast/", ForecastAPI.as_view(), name="forecast_api"),
]
