# Generated by Django 4.2.20 on 2025-03-16 01:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='analysishistory',
            name='similarity_score',
            field=models.FloatField(blank=True, null=True, verbose_name='相似度得分'),
        ),
    ]
