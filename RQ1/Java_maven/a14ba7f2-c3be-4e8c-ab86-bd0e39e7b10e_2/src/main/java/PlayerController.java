public class PlayerController : MonoBehaviour
{
    public Inventory inventory;

    private void OnTriggerEnter2D(Collider2D collision)
    {
        ItemData itemData = collision.gameObject.GetComponent<ItemData>();
        if (itemData != null)
        {
            inventory.AddItem(itemData);
            Destroy(collision.gameObject);
        }
    }
}
