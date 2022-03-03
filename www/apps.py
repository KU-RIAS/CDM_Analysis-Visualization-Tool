from django.apps import AppConfig


class WwwConfig(AppConfig):

    def ready(self):
        DATABASES = {}
        # Your database connections
