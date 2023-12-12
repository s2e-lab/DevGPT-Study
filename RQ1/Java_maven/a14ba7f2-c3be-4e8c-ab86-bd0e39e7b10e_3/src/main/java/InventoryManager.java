public class InventoryManager : MonoBehaviour
{
    public GameObject inventoryPanel;
    public GameObject inventorySlotPrefab;

    private List<Item> inventoryItems = new List<Item>();
    private List<GameObject> inventorySlots = new List<GameObject>();

    public void OnDrop(PointerEventData eventData)
    {
        SmallInventorySlot smallSlot = eventData.pointerDrag.GetComponent<SmallInventorySlot>();

        if (smallSlot != null && smallSlot.item != null)
        {
            AddItem(smallSlot.item);
            smallSlot.SetItem(null);
            OnInventoryChanged();
        }
    }

    // Other methods for managing the hidden inventory...
}
