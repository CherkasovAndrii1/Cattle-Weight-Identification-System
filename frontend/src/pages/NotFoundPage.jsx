import React from 'react';
import { Link } from 'react-router-dom';

function NotFoundPage() {
  return (
    <div>
      <h1>404 - Not Found</h1>
      <p>Request page doesn't seem to be developed yet.</p>
      <Link to="/">Back home</Link>
    </div>
  );
}
export default NotFoundPage;