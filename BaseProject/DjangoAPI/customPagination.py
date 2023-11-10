from rest_framework import pagination
from rest_framework.response import Response

class CustomPagination(pagination.PageNumberPagination):
    page_size = 20
    def get_page_number(self, data):
        return Response({
            'navigation': {
                'next_page': self.get_next_link(),
                'previous- _page': self.get_previous_link()
            },
            'page_count': self.page.paginator.count,
            'current_page': self.page.number,
            'results': data
        })