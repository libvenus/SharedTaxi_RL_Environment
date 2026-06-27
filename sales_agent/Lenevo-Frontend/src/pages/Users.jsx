import Card from "../components/Card";
import PageTitle from "../components/PageTitle";
import Button from "../components/Button";

function Users() {
  return (
    <>
      <PageTitle
        title="Users"
        subtitle="Manage Users"
      />

      <Card>
        <h3>User List</h3>

        <Button
          text="Add User"
          type="primary"
          onClick={() => alert("Add User")}
        />

        <Button
          text="Delete User"
          type="danger"
          onClick={() => alert("Delete User")}
        />
      </Card>
    </>
  );
}

export default Users;