"""
URL configuration for bin2src_web project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from core import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('captcha/', include('captcha.urls')),
    path('', views.login_view, name='login'),
    path('login/', views.login_view, name='login_alt'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('binary-compare/', views.binary_compare_view, name='binary_compare'),
    path('vulnerability-analysis/', views.vulnerability_analysis_view, name='vulnerability_analysis'),
    path('change-password/', views.change_password_view, name='change_password'),
    path('delete-history/<int:history_id>/', views.delete_history_view, name='delete_history'),
    path('analysis/<int:analysis_id>/', views.analysis_detail_view, name='analysis_detail'),
    
    # 管理员URL
    path('admin-panel/', views.admin_dashboard, name='admin_dashboard'),
    path('admin-panel/user/<int:user_id>/analyses/', views.admin_user_analyses, name='admin_user_analyses'),
    path('admin-panel/user/<int:user_id>/reset-password/', views.admin_reset_user_password, name='admin_reset_user_password'),
    path('admin-panel/upload-vulnerability/', views.admin_upload_vulnerability_file, name='admin_upload_vulnerability'),
    path('admin-panel/update-model/', views.admin_update_model, name='admin_update_model'),
    path('admin-panel/delete-vulnerability/<str:file_name>/', views.admin_delete_vulnerability_file, name='admin_delete_vulnerability'),

     # API路由
    path('api/cwe/<int:cwe_number>/', views.get_cwe_info, name='get_cwe_info'), 
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
