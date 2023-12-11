from django.db.models import Count

# assuming your models are named Post and Comment

posts = Post.objects.annotate(comment_count=Count('comment'))

for post in posts:
    print(f"Post {post.id} has {post.comment_count} comments")
