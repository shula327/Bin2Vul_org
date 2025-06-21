from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
import json

# Create your models here.

class AnalysisHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name='用户')
    analysis_type = models.CharField(max_length=20, choices=[
        ('binary_compare', '二进制代码比对'),
        ('vulnerability', '漏洞分析')
    ], verbose_name='分析类型')
    file1 = models.FileField(upload_to='uploads/%Y/%m/%d/', verbose_name='文件1')
    file2 = models.FileField(upload_to='uploads/%Y/%m/%d/', blank=True, null=True, verbose_name='文件2')
    similarity_score = models.FloatField(verbose_name='相似度得分', null=True, blank=True)
    remarks = models.TextField(verbose_name='备注', blank=True, null=True, help_text='存储漏洞类型或其他分析结果')
    top_results = models.TextField(verbose_name='Top相似结果', blank=True, null=True, help_text='JSON格式存储的前K个最相似结果')
    created_at = models.DateTimeField(default=timezone.now, verbose_name='创建时间')
    
    class Meta:
        verbose_name = '分析历史'
        verbose_name_plural = '分析历史'
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.user.username} - {self.analysis_type} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
        
    def get_top_results(self):
        """获取解析后的Top结果列表"""
        if not self.top_results:
            return []
        try:
            return json.loads(self.top_results)
        except:
            return []
            
    def get_similarity_percentage(self):
        """获取百分比形式的相似度"""
        if self.similarity_score is None:
            return None
        # 将[-1,1]范围的余弦相似度映射到[0,100]的百分比
        return (self.similarity_score + 1) * 50


class SystemConfig(models.Model):
    """系统配置模型，用于存储全局配置项"""
    key = models.CharField(max_length=50, unique=True, verbose_name='配置键')
    value = models.CharField(max_length=255, verbose_name='配置值')
    description = models.TextField(blank=True, null=True, verbose_name='配置说明')
    updated_at = models.DateTimeField(auto_now=True, verbose_name='更新时间')
    
    class Meta:
        verbose_name = '系统配置'
        verbose_name_plural = '系统配置'
        ordering = ['key']
    
    def __str__(self):
        return f"{self.key}: {self.value}"
    
    @classmethod
    def get_value(cls, key, default=None):
        """获取配置值，如不存在则返回默认值"""
        try:
            return cls.objects.get(key=key).value
        except cls.DoesNotExist:
            return default
    
    @classmethod
    def set_value(cls, key, value, description=None):
        """设置配置值，不存在则创建"""
        obj, created = cls.objects.update_or_create(
            key=key,
            defaults={'value': value, 'description': description}
        )
        return obj
