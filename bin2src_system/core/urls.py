from django.urls import path
from . import views
from .views import AnalysisDetailView, AnalysisDeleteView

urlpatterns = [
    # 已有的URL路径保持不变...
    
    # 分析历史列表
    path('analysis-history/', views.analysis_history, name='analysis_history'),
    
    # 添加新的URL路径
    # path('analysis/<int:pk>/', AnalysisDetailView.as_view(), name='analysis_detail'),  # 注释掉原来的
    path('analysis/<int:analysis_id>/', views.analysis_detail_view, name='analysis_detail'),  # 使用新的函数视图
    path('analysis/<int:pk>/delete/', AnalysisDeleteView.as_view(), name='analysis_delete'),
    #path('delete-history/<int:history_id>/', views.delete_history_view, name='delete_history'),
] 