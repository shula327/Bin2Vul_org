# Generated by Django 4.2.20 on 2025-03-21 11:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0002_alter_analysishistory_similarity_score'),
    ]

    operations = [
        migrations.AddField(
            model_name='analysishistory',
            name='description',
            field=models.TextField(blank=True, null=True, verbose_name='分析描述'),
        ),
        migrations.AddField(
            model_name='analysishistory',
            name='result_image',
            field=models.ImageField(blank=True, null=True, upload_to='analysis_results/%Y/%m/%d/', verbose_name='分析结果图像'),
        ),
    ]
