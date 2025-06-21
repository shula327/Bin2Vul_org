from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # 已有的URL路径...
]

# 在开发环境中添加媒体文件服务
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT) 