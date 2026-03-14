from rest_framework import serializers


class ForecastRequestSerializer(serializers.Serializer):
    data = serializers.ListField(
        child=serializers.DictField(),
        required=False,
        allow_empty=False,
    )
    file = serializers.FileField(required=False)
    steps = serializers.IntegerField(min_value=1, default=24)
