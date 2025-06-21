from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordChangeForm
from django.contrib.auth.models import User
from captcha.fields import CaptchaField

class CustomUserCreationForm(UserCreationForm):
    email = forms.EmailField(required=True, widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': '请输入邮箱'}))
    
    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            if field.label:
                field.widget.attrs['placeholder'] = f'请输入{field.label}'

class CustomAuthenticationForm(AuthenticationForm):
    captcha = CaptchaField(label='验证码', error_messages={'invalid': '验证码错误，请重新输入'})
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            if field.label:
                field.widget.attrs['placeholder'] = f'请输入{field.label}'

class CustomPasswordChangeForm(PasswordChangeForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs['class'] = 'form-control'
            if field.label:
                field.widget.attrs['placeholder'] = f'请输入{field.label}'

class BinaryCompareForm(forms.Form):
    file1 = forms.FileField(label='二进制文件1', widget=forms.FileInput(attrs={
        'class': 'form-control',
        'accept': '.elf,.exe,.dll,.so,.dylib'
    }))
    file2 = forms.FileField(label='二进制文件2', widget=forms.FileInput(attrs={
        'class': 'form-control',
        'accept': '.elf,.exe,.dll,.so,.dylib'
    }))


class VulnerabilityAnalysisForm(forms.Form):
    file = forms.FileField(label='二进制文件', widget=forms.FileInput(attrs={
        'class': 'form-control',
        'accept': '.elf,.exe,.dll,.so,.dylib'
    })) 
