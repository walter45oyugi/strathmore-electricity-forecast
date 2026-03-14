from django import forms


class UploadFileForm(forms.Form):
    file = forms.FileField(
        label="Select a CSV or Excel file",
        widget=forms.ClearableFileInput(
            attrs={"accept": ".csv,.xls,.xlsx"}
        ),
    )
    steps = forms.IntegerField(
        label="Forecast horizon (hours)",
        min_value=1,
        initial=24,
    )